"""Exercise real Agate adapters through the authenticated Runtime HTTP boundary."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from atrex_gateway_client import build_eval_request_from_content
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import (
    FakeAgateClient,
    FakeGatewayError,
    _adapter,
    _contract,
    _exploratory_request,
)
from test_agent_abba import FakeContexts, NativeFakeAgate
from test_gateway_abba import _abba_value
from test_gateway_private_results import _nested_log_tail
from test_gateway_proxy import _request, _service

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.agent_abba import AgentAbbaGatewayAdapter
from atrex_runtime.gateway.contract import AgateEvaluationContext
from atrex_runtime.gateway.failures import infrastructure_detail


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


def _assert_diagnostic(text: str) -> None:
    for expected in (
        "error_class", "code_execution_failed", "python_exception", "trace-candidate-failure",
        "candidate/kernel.py", "baseline kernel supports page_size == 32",
    ):
        assert expected in text
    for hidden in (
        "private-stdout", "private-stderr", "private-reference", "private-input",
        "151904033", "manual_seed", "atrex_bench/eval", "site-packages", "/tmp/agate-",
    ):
        assert hidden not in text


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["full", "correctness_only"])
@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_evaluate_job_diagnostic_reaches_http_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, status: str,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient(_failed_job(status))
    adapter, _, jobs = _adapter(tmp_path, client)
    monkeypatch.setattr(service, "_adapter", adapter)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    request = json.loads(_request(attempt))
    if mode == "correctness_only":
        request["mode"] = mode
        request.pop("latency_prediction")
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            # A failed measurement remains retryable rather than cached as a verdict.
            for _ in range(2):
                response = await http.post(
                    "/v1/operations", json=request,
                    headers={"Authorization": f"Bearer {capability.token}"},
                )
                assert response.status_code == 503
                body = response.json()
                assert body["error"] == "gateway_unavailable"
                _assert_diagnostic(body["detail"])
                assert "ev_test" in body["detail"]
                assert status in body["detail"]
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
            # The repetition TaskGroup must not hide the nested, actionable exception.
            with pytest.raises(InfrastructureError) as raised:
                await adapter.execute(request)
            _assert_diagnostic(infrastructure_detail(raised.value))
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
