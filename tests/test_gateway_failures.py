"""Gateway failures remain diagnosable without exposing credentials or private cases."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from test_gateway_proxy import _request, _service

from atrex_runtime.domain.errors import InfrastructureError, UpstreamGatewayError
from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.abba import _parse_remote_payload
from atrex_runtime.gateway.abba_remote import RESULT_PREFIX
from atrex_runtime.gateway.failures import infrastructure_detail
from atrex_runtime.gateway.source_tree import SourceTreeAgateClient


@pytest.mark.anyio
async def test_proxy_records_cause_returns_detail_and_allows_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    original = adapter.execute

    async def broken(_request):
        try:
            raise RuntimeError("git fetch: upload-pack: not our ref deadbeef")
        except RuntimeError as cause:
            raise InfrastructureError("Evaluator Bundle could not be loaded") from cause

    monkeypatch.setattr(adapter, "execute", broken)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(64 * 1024, 8, 16 * 1024))
    sent = []

    async def receive():
        return {"type": "http.request", "body": _request(attempt), "more_body": False}

    async def send(message):
        sent.append(message)

    with caplog.at_level(logging.ERROR):
        await app({
            "type": "http", "method": "POST", "path": "/v1/operations",
            "headers": [(b"authorization", f"Bearer {capability.token}".encode())],
        }, receive, send)
    assert sent[0]["status"] == 503
    assert json.loads(sent[1]["body"]) == {
        "error": "gateway_unavailable", "detail": "Evaluator Bundle could not be loaded",
    }
    assert "not our ref deadbeef" in caplog.text
    assert str(attempt.id) in caplog.text
    assert capability.token not in caplog.text
    failed = [event for event in registry.list_runtime_events(after_sequence=0, limit=100)
              if event.kind == "gateway.operation_failed"]
    assert failed[-1].payload["detail"] == "Evaluator Bundle could not be loaded"
    assert failed[-1].payload["error_type"] == "InfrastructureError"
    # An infrastructure response must not become a cached authoritative result.
    monkeypatch.setattr(adapter, "execute", original)
    recovered = await service.execute(capability.token, _request(attempt))
    assert recovered.status == "completed"
    control.close()
    registry.close()


def test_public_detail_omits_credentials_and_is_bounded(monkeypatch):
    monkeypatch.setenv("AGATE_SK", "fixture-secret-value")
    error = InfrastructureError(
        "Fetch https://user:password@host/repo failed; key=fixture-secret-value; "
        "Authorization: Bearer fixture-token; " + "字" * 10000
    )
    detail = infrastructure_detail(error)
    for secret in ("password", "fixture-secret-value", "fixture-token"):
        assert secret not in detail
    assert "Fetch https://[REDACTED]@host/repo failed" in detail
    assert len(detail.encode()) <= 8192


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/v1/operations", "/v1/runtime/queries", "/v1/runtime/journals"])
@pytest.mark.parametrize("upstream", [False, True])
async def test_http_boundary_logs_failures_before_operation_dispatch(path, upstream, caplog):
    class Service:
        async def execute(self, *args, **kwargs):
            if upstream:
                raise UpstreamGatewayError(503, "upstream scheduler unavailable")
            raise InfrastructureError("artifact storage unavailable")

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        sent.append(message)

    app = GatewayProxyAsgiApp(Service(), GatewayProxyLimits(65536, 8, 16384))
    with caplog.at_level(logging.ERROR):
        await app({"type": "http", "method": "POST", "path": path,
                   "headers": [(b"authorization", b"Bearer test-capability")]}, receive, send)
    detail = "upstream scheduler unavailable" if upstream else "artifact storage unavailable"
    assert sent[0]["status"] == 503
    assert json.loads(sent[1]["body"])["detail"] == detail
    assert detail in caplog.text
    assert "test-capability" not in caplog.text


@pytest.mark.parametrize("raw", [
    {"raw_result": True, "error": "hidden input n=987654321", "runs": []},
    {"raw_result": True, "runs": [{"result": {"private_inputs": [987654321]}}]},
])
def test_source_tree_private_failure_has_an_actionable_public_detail(raw):
    class Client:
        def get_job(self, job_id, **kwargs):
            return {"job_id": job_id, "status": "succeeded",
                    "result": {"stdout": RESULT_PREFIX + json.dumps(raw)}}

    try:
        SourceTreeAgateClient(Client(), None).get_job("dv_failed")
    except InfrastructureError as cause:
        error = InfrastructureError(f"Agate SDK request failed: {cause}")
        error.__cause__ = cause
    else:
        pytest.fail("Malformed source-tree result should fail")
    assert "987654321" in str(error)  # Full diagnostic remains available to trusted logs.
    detail = infrastructure_detail(error)
    assert "987654321" not in detail
    assert "dv_failed" in detail
    assert "Runtime logs" in detail


def test_abba_private_driver_error_stays_in_trusted_logs():
    job = {
        "job_id": "dv_abba_failed",
        "result": {"stdout": RESULT_PREFIX + json.dumps({
            "schema_version": 1, "error": "hidden input n=987654321",
        })},
    }
    with pytest.raises(InfrastructureError) as raised:
        _parse_remote_payload(job, [])
    assert "987654321" in str(raised.value)
    detail = infrastructure_detail(raised.value)
    assert "987654321" not in detail
    assert "dv_abba_failed" in detail
    assert "Runtime logs" in detail
