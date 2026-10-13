"""Exercise real Agate adapters through the authenticated Runtime HTTP boundary."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import test_kernel_sources
from atrex_gateway_client import build_eval_request_from_content
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import (
    FakeAgateClient,
    FakeGatewayError,
    _adapter,
    _contract,
    _exploratory_request,
    _successful_job,
)
from test_agent_abba import FakeContexts, NativeFakeAgate
from test_gateway_abba import _abba_value
from test_gateway_private_results import _nested_log_tail
from test_gateway_proxy import _request, _service

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway import GatewayOperation, GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.abba_remote import RESULT_PREFIX
from atrex_runtime.gateway.agent_abba import AgentAbbaGatewayAdapter
from atrex_runtime.gateway.contract import AgateEvaluationContext
from atrex_runtime.gateway.source_tree import SourceTreeAgateClient

source_seed = test_kernel_sources.source_seed


def _failed_job(status: str = "failed") -> dict:
    return {
        "job_id": "ev_candidate_failure",
        "status": status,
        "error": {
            "error_class": "code",
            "reason": "code_execution_failed",
            "trace_id": "trace-candidate-failure",
            "details": {
                "failure_origin": "code",
                "failure_rule": "python_exception",
                "logs_tail": _nested_log_tail(),
            },
        },
        "result": None,
        "stdout": "private-stdout",
        "stderr": "private-stderr",
        "spec": {"reference_py": "private-reference", "input_py": "private-input"},
    }


def _assert_diagnostic(text: str, *, include_error_class: bool = True) -> None:
    for expected in (
        "error_class", "code_execution_failed", "python_exception", "trace-candidate-failure",
        "candidate/kernel.py", "baseline kernel supports page_size == 32",
    ):
        if expected == "error_class" and not include_error_class:
            continue
        assert expected in text
    for hidden in (
        "private-stdout", "private-stderr", "private-reference", "private-input",
        "151904033", "manual_seed", "atrex_bench/eval", "site-packages", "/tmp/agate-",
    ):
        assert hidden not in text


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only"])
@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("error_class", ["code", "compile", "unknown", None])
async def test_evaluate_job_diagnostic_reaches_http_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, status: str,
    error_class: str | None,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    job = _failed_job(status)
    if error_class is None:
        job["error"].pop("error_class")
    else:
        job["error"]["error_class"] = error_class
    client = FakeAgateClient(job)
    adapter, _, jobs = _adapter(tmp_path, client)
    monkeypatch.setattr(service, "_adapter", adapter)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    request = json.loads(_request(attempt))
    if mode == "correctness_only":
        request["mode"] = mode
        request.pop("latency_prediction")
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            first = None
            for _ in range(2):
                response = await http.post(
                    "/v1/operations", json=request,
                    headers={"Authorization": f"Bearer {capability.token}"},
                )
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["status"] == status
                assert body["result"]["correct"] is None
                assert body["result"]["correctness"] is None
                assert body["result"]["latency_us_geomean"] is None
                _assert_diagnostic(response.text, include_error_class=error_class is not None)
                assert "ev_test" in response.text
                if error_class is not None:
                    assert error_class in response.text
                assert not control.list_evaluations(attempt.id)
                if first is None:
                    first = body
                    submitted = len(client.submitted)
                else:
                    assert body == first
                    assert len(client.submitted) == submitted
            read = await http.post(
                "/v1/runtime/queries", json={
                    "schema_version": 2, "attempt_id": attempt.id,
                    "idempotency_key": "read-terminal-failure",
                    "operation": "result_artifact_read",
                    "result_artifact_digest": first["result_artifact_digest"],
                },
                headers={"Authorization": f"Bearer {capability.token}"},
            )
            assert read.status_code == 200, read.text
            assert read.json()["result"]["result"] == first["result"]
            # A fresh request may retry exactly the same source after a terminal
            # execution failure. The stored failure is never a Kernel verdict.
            client.job = _successful_job()
            response = await http.post(
                "/v1/operations", json={**request, "idempotency_key": "retry-same-source"},
                headers={"Authorization": f"Bearer {capability.token}"},
            )
            assert response.status_code == 200, response.text
            assert response.json()["result"]["correct"] is True
            assert len(client.submitted) > submitted
            assert len(control.list_evaluations(attempt.id)) == (1 if mode == "full" else 0)
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_evaluate_transport_failure_still_returns_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient(_successful_job(), submit_error=FakeGatewayError(503, "infra", {}))
    adapter, _, jobs = _adapter(tmp_path, client)
    monkeypatch.setattr(service, "_adapter", adapter)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post(
                "/v1/operations", json=json.loads(_request(attempt)),
                headers={"Authorization": f"Bearer {capability.token}"},
            )
        assert response.status_code == 503
        assert response.json()["error"] == "gateway_unavailable"
        assert not control.list_evaluations(attempt.id)
        assert control.get_operation_artifact(
            attempt.id, "evaluate-candidate-1", GatewayOperation.EVALUATE,
        ) is None
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_successful_sibling_batch_cannot_mask_execution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)

    class Client(FakeAgateClient):
        def get_job(self, job_id, **kwargs):
            request = self.requests_by_job[job_id]
            if "1" in request["reference"]["shapes"]:
                return {**_failed_job(status), "job_id": job_id}
            return super().get_job(job_id, **kwargs)

    client = Client(_successful_job())
    adapter, _, jobs = _adapter(tmp_path, client)
    monkeypatch.setattr(service, "_adapter", adapter)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post(
                "/v1/operations", json=json.loads(_request(attempt)),
                headers={"Authorization": f"Bearer {capability.token}"},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == status
        assert body["result"]["correct"] is None
        assert body["result"]["latency_us_by_shape"] == {}
        assert not control.list_evaluations(attempt.id)
        details = body["result"]["error"]["details"]
        assert len(details) == 1 and details[0]["batch_index"] == 1
        _assert_diagnostic(response.text)
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["job", "rejection"])
async def test_adapter_repetitions_preserve_failure_diagnostics(
    tmp_path: Path, failure: str,
) -> None:
    rejection = FakeGatewayError(422, "validation", {
        "reason": "candidate_validation_failed",
        "details": {"forbidden_imports": ["subprocess"], "input_py": "private-input"},
    })
    client = FakeAgateClient(
        _failed_job(), submit_error=rejection if failure == "rejection" else None,
    )
    adapter, _, jobs = _adapter(tmp_path, client)
    adapter._optimizer_evaluate_repeats = 2
    request = _exploratory_request(tmp_path, {})
    try:
        if failure == "job":
            result = await adapter.execute(request)
            assert result.status == "failed" and result.evaluation is None
            assert result.worker_result["correct"] is None
            _assert_diagnostic(json.dumps(result.worker_result))
        else:
            result = await adapter.execute(request)
            assert result.evaluation.correct is False
            errors = result.worker_result["error"]["details"]
            assert len(errors) == 2
            for error in errors:
                rendered = json.dumps(error)
                assert "candidate_validation_failed" in rendered
                assert "subprocess" in rendered
                assert "private-input" not in rendered
    finally:
        jobs.close()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only"])
@pytest.mark.parametrize("failure", ["driver", "child", "stderr", "exit", "secret", "compiler"])
async def test_source_tree_inner_failure_reaches_http_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_seed, mode: str, failure: str,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    raw = {"raw_result": True, "error": None, "runs": []}
    expected = {
        "driver": "required GPU tool is not installed: ncu",
        "child": "No module named 'cutlass'",
        "stderr": "baseline kernel supports page_size == 32",
        "exit": '"exit_code": 7',
        "secret": "No module named 'cutlass'",
        "compiler": 'kernel.cu(12): error: identifier "bad_symbol" is undefined',
    }
    monkeypatch.setenv("AGATE_SK", "fixture-source-secret")
    if failure in {"driver", "secret"}:
        raw["error"] = (
            "RuntimeError: required GPU tool is not installed: ncu" if failure == "driver"
            else "ModuleNotFoundError: No module named 'cutlass'; key=fixture-source-secret; "
            "Authorization: Bearer fixture-source-token"
        )
    else:
        raw["runs"] = [{
            "exit_code": 7 if failure == "exit" else 0,
            "result": (
                {"raw_result": _successful_job()["result"]} if failure == "exit"
                else {"error": "ModuleNotFoundError: No module named 'cutlass'"}
                if failure == "child" else None
            ),
            "stderr_tail": (
                _nested_log_tail() if failure == "stderr"
                else '/tmp/agate-private/candidate/kernel.cu(12): error: '
                'identifier "bad_symbol" is undefined\nprivate-input: 151904033'
                if failure == "compiler" else ""
            ),
            "stdout_tail": "private-stdout",
            "input_kwargs": {"private-input": 151904033},
        }]
    job = {
        "status": "succeeded", "command_ok": True,
        "result": {"stdout": RESULT_PREFIX + json.dumps(raw), "stderr": "private-stderr"},
    }
    client = FakeAgateClient(job)
    adapter, _, jobs = _adapter(tmp_path, client)
    adapter._client = SourceTreeAgateClient(client, None)
    adapter._contexts.context = replace(
        adapter._contexts.context,
        contract=_contract().model_copy(
            update={"kernel_sources": {Dsl.TRITON: source_seed.contract}},
        ),
    )
    monkeypatch.setattr(service, "_adapter", adapter)
    request = json.loads(_request(attempt))
    request["candidate"]["files"] = [
        {"path": path, "content_base64": base64.b64encode(text.encode()).decode()}
        for path, text in source_seed.contract.validate_tree(source_seed.working).items()
    ]
    if mode == "correctness_only":
        request["mode"] = mode
        request.pop("latency_prediction")
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post(
                "/v1/operations", json=request,
                headers={"Authorization": f"Bearer {capability.token}"},
            )
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        if failure == "compiler":
            diagnostic = json.loads(detail[detail.index("{"):])
            assert diagnostic["runs"][0]["compiler_errors"] == [expected[failure]]
        else:
            assert expected[failure] in detail
        assert "ev_test" in detail
        assert not control.list_evaluations(attempt.id)
        assert {kind for kind, _ in client.submitted} == {"eval"}
        for hidden in ("private-input", "private-stdout", "private-stderr", "151904033",
                       "manual_seed", "atrex_bench/eval", "site-packages", "/tmp/agate-",
                       "fixture-source-secret", "fixture-source-token"):
            assert hidden not in detail
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure", ["job", "cancelled", "command", "schedule", "sdk_evidence", "exit"],
)
async def test_abba_batch_diagnostic_reaches_http_agent_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    registry, control, attempt, capability, service, delegate = _service(tmp_path)

    class Client(NativeFakeAgate):
        def get_job(self, job_id, **kwargs):
            job = super().get_job(job_id, **kwargs)
            if failure in {"job", "cancelled"}:
                job = {**deepcopy(_failed_job()), "job_id": job_id}
                if failure == "cancelled":
                    job["status"] = "cancelled"
            elif failure == "command":
                # Outer success is not permission to expose raw result/log payloads.
                job["command_ok"] = False
                job["error"] = _failed_job()["error"]
                job["result"]["extra"] = "private-result"
            elif failure == "exit":
                job["result"]["exit_code"] = 7
            return job

    client = Client()
    client.failure = failure if failure in {"schedule", "sdk_evidence"} else None
    contexts = FakeContexts(AgateEvaluationContext("vector_add", "H20", Dsl.TRITON, _contract()))
    adapter = AgentAbbaGatewayAdapter(
        delegate, client, contexts, LocalArtifactStore(tmp_path / "artifacts"),
        None, build_eval_request_from_content, wait_timeout_s=90,
    )
    monkeypatch.setattr(service, "_adapter", adapter)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    expected = {
        "schedule": "native Agate ABBA run order differs from the request",
        "sdk_evidence": "native Agate ABBA result has incomplete SDK evidence",
        "exit": "exit_code=7",
    }
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            first = None
            for _ in range(2):
                response = await http.post(
                    "/v1/operations", json=_abba_value(attempt),
                    headers={"Authorization": f"Bearer {capability.token}"},
                )
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["status"] == "failed"
                assert body["result"]["correct"] is False
                assert body["result"]["speedup"] is None
                error = body["result"]["error"]
                assert error["category"] == "abba_execution_failed"
                details = error["details"]
                assert len(details) == 2
                assert {item["batch_index"] for item in details} == {0, 1}
                assert all(item["job_id"].startswith("ev_agent_abba_") for item in details)
                rendered = json.dumps(details)
                if failure in {"job", "cancelled", "command"}:
                    _assert_diagnostic(rendered)
                else:
                    assert expected[failure] in rendered
                assert "private-input-reference-or-log-must-not-escape" not in rendered
                assert "private-result" not in rendered
                if first is None:
                    first = body["result"]
                    submitted = len(client.requests)
                else:
                    assert body["result"] == first
                    assert len(client.requests) == submitted
    finally:
        control.close()
        registry.close()
