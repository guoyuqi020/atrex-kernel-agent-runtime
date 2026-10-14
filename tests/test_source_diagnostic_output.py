"""Real diagnostic driver -> bounded Dev stdout -> Runtime HTTP -> Agent view."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import test_source_diagnostics
from httpx import ASGITransport, AsyncClient
from test_gateway_proxy import _request, _service

from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.source_diagnostics import (
    DIAGNOSTIC_PREFIX,
    LOG_TEXT_LIMIT,
    RESPONSE_LIMIT,
    _encode_result,
    _export,
)

diagnostic = test_source_diagnostics.diagnostic
source_seed = test_source_diagnostics.source_seed


def _candidate_request(attempt, diagnostic, operation, parameters):
    value = json.loads(_request(attempt))
    value.pop("latency_prediction")
    value.update(operation=operation, **parameters)
    value["candidate"]["files"] = [
        {"path": name, "content_base64": base64.b64encode(text.encode()).decode()}
        for name, text in diagnostic.context.kernel_source.validate_tree(
            diagnostic.seed.working,
        ).items()
    ]
    return value


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["disassemble", "profile", "check"])
async def test_large_tool_output_keeps_summary_and_truncation_reason_through_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic,
    operation: str,
):
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    monkeypatch.setattr(service, "_adapter", diagnostic.adapter)
    parameters = {"env_vars": {"FAKE_LARGE_EXPORTS": "1"}}
    if operation == "profile":
        parameters.update(level="survey", source=True)
    if operation == "check":
        parameters.update(sanitize="memcheck")
        parameters["env_vars"]["FAKE_SANITIZER_ERROR"] = "1"
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 16, 32768))
    headers = {"Authorization": f"Bearer {capability.token}"}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post(
                "/v1/operations",
                headers=headers,
                json=_candidate_request(attempt, diagnostic, operation, parameters),
            )
            assert response.status_code == 200, response.text[:2000]
            body = response.json()
            # Read the durable Agent-facing artifact too, not only the live HTTP result.
            read = await http.post(
                "/v1/runtime/queries",
                headers=headers,
                json={
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": "read-diagnostic",
                    "operation": "result_artifact_read",
                    "result_artifact_digest": body["result_artifact_digest"],
                },
            )
        assert read.status_code == 200, read.text[:2000]
        diagnostic_result = body["result"]["result"]
        assert diagnostic_result["operation"] == operation
        assert diagnostic_result["shape_id"] == "0"
        assert diagnostic_result["passed"] is (operation != "check")
        summary = diagnostic_result["output_summary"]
        assert summary["truncated"] is True
        assert summary["reason"] in {"diagnostic_text_size_limit", "diagnostic_response_size_limit"}
        assert "not a Kernel failure" in summary["message"]
        assert summary["response_limit_bytes"] == RESPONSE_LIMIT
        assert read.json()["result"]["result"] == body["result"]
        for returned in (response.text, read.text):
            assert "PRIVATE_INPUT" not in returned
            assert '"logs"' not in returned
        assert not control.list_evaluations(attempt.id)
        assert len(diagnostic.sdk.requests) == 1
        raw = diagnostic.sdk.jobs["dv_0"]["result"]
        assert raw["output_truncated"] is False
        assert raw["output_bytes"] <= RESPONSE_LIMIT < 1048576
        assert len(response.content) < 1048576
        record = json.loads(raw["stdout"].removeprefix(DIAGNOSTIC_PREFIX))
        if operation == "check":
            assert diagnostic_result["failure_stage"] == "execution"
            assert diagnostic_result["sanitizer_passed"] is False
            assert diagnostic_result["error"] == "compiler, GPU tool or candidate execution failed"
            assert record["logs"][0]["stdout"]["size_bytes"] > 1048576
            assert len(record["logs"][0]["stdout"]["text"].encode()) <= LOG_TEXT_LIMIT
        else:
            assert diagnostic_result["kernels"][0]["duration_us"] == 2
            assert summary["reason"] == "diagnostic_response_size_limit"
            assert record["exports"]["sass.txt"]["size_bytes"] > 1048576
            assert record["exports"]["sass.txt"]["truncated"] is True
            for log in record["logs"]:
                assert "text" not in log["stdout"]
                assert log["stdout"]["content_omitted"] is True
            # The full assembly is fingerprinted once, not duplicated in logs.
            assert record["logs"][-1]["stdout"]["sha256"] == record["exports"]["sass.txt"]["sha256"]
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["disassemble", "profile", "check"])
@pytest.mark.parametrize("truncated", [False, True])
async def test_legacy_missing_or_truncated_summary_is_an_explicit_transport_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic,
    operation: str,
    truncated: bool,
):
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    monkeypatch.setattr(service, "_adapter", diagnostic.adapter)
    # Recreate the old driver: SASS once in exports and again JSON-escaped in
    # execution logs. Agate retains the tail and discards the record prefix.
    assembly = "/*0010*/ MOV R1, R2;\n" * 26000
    legacy = (
        DIAGNOSTIC_PREFIX
        + json.dumps(
            {
                "operation": operation,
                "passed": True,
                "exports": {"sass.txt": {"text": assembly}},
                "logs": [{"text": json.dumps({"stdout": assembly, "stderr": "PRIVATE_INPUT"})}],
            }
        )
        + "\n"
    )
    if truncated:
        assert len(legacy.encode()) > 1048576
        stdout = legacy.encode()[-1048576:].decode()
        assert DIAGNOSTIC_PREFIX not in stdout
    else:
        stdout = DIAGNOSTIC_PREFIX + '{"operation": "disassemble", "PRIVATE_INPUT": '

    def get_job(job_id, **kwargs):
        return {
            "job_id": job_id,
            "kind": "dev",
            "status": "succeeded",
            "result": {
                "stdout": stdout,
                "exit_code": 0,
                "output_bytes": len(legacy.encode()),
                "output_truncated": truncated,
            },
        }

    monkeypatch.setattr(diagnostic.sdk, "get_job", get_job)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 16, 32768))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            response = await http.post(
                "/v1/operations",
                headers={"Authorization": f"Bearer {capability.token}"},
                json=_candidate_request(attempt, diagnostic, operation, {}),
            )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "not" in detail and "Kernel failure" in detail
        assert "PRIVATE_INPUT" not in response.text
        if truncated:
            assert "diagnostic_output_truncated" in detail
            assert f"original_output_bytes={len(legacy.encode())}" in detail
            assert "retained_stdout_bytes=1048576" in detail
            assert "Reading the same completed job again cannot restore" in detail
        else:
            assert "missing, malformed" in detail
            assert "diagnostic_output_truncated" not in detail
        assert not control.list_evaluations(attempt.id)
    finally:
        control.close()
        registry.close()


def test_large_counter_metadata_cannot_displace_diagnostic_verdict():
    result = {
        "operation": "profile",
        "shape_id": "0",
        "passed": True,
        "status": "passed",
        "kernels": [
            {
                "name": "kernel",
                "duration_us": 2,
                "compute_sol_pct": 60,
                "metrics": {"large": "x" * (2 * RESPONSE_LIMIT)},
            }
        ],
    }
    text = _encode_result(result)
    assert len(text.encode()) <= RESPONSE_LIMIT
    decoded = json.loads(text.removeprefix(DIAGNOSTIC_PREFIX))
    assert decoded["passed"] is True and decoded["kernels"][0]["duration_us"] == 2
    assert decoded["kernels"][0]["compute_sol_pct"] == 60
    assert decoded["output_summary"]["truncated_fields"] == ["kernels.metrics"]
    assert "metrics" in result["kernels"][0]  # The unbounded input remains unchanged.


def test_serialized_budget_counts_escaping_and_preserves_fingerprints():
    raw = '\\"\n\x01中文' * 100000
    export = _export(raw)
    text = _encode_result(
        {
            "operation": "disassemble",
            "passed": True,
            "status": "passed",
            "shape_id": "0",
            "exports": {"sass.txt": export, "metrics.csv": _export(raw)},
        }
    )
    assert len(text.encode()) <= RESPONSE_LIMIT
    decoded = json.loads(text.removeprefix(DIAGNOSTIC_PREFIX))
    assert decoded["passed"] is True
    for item in decoded["exports"].values():
        assert item["size_bytes"] == len(raw.encode())
        assert item["sha256"] == export["sha256"]
        assert item["truncated"] is True
    assert decoded["output_summary"]["reason"] == "diagnostic_response_size_limit"
