"""Long-poll expiry must not publish unfinished Agent diagnostic/Dev results."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import FakeAgateClient, _adapter
from test_gateway_proxy import _request, _service

from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.gateway import GatewayOperation, GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.proxy import GatewayAdapterResult

OPERATIONS = ("profile", "check", "disassemble", "dev")


def _job_request(attempt, operation: str) -> dict:
    request = json.loads(_request(attempt))
    request.update(operation=operation, idempotency_key="job-wait")
    if operation == "profile":
        request["level"] = "sol"
    elif operation == "dev":
        request["command"] = "python3 kernel.py"
    request.pop("latency_prediction")
    return request


def _terminal_job(status: str, operation: str) -> dict:
    results = {
        "profile": {"kernels": [{"name": "vector_add", "duration_us": 4.0}]},
        "check": {"compile_ok": True, "sanitizer_passed": True},
        "disassemble": {"format": "ptx", "assembly": ".entry candidate() { ret; }"},
        "dev": {"stdout": "probe result: 42\n", "stderr": "probe diagnostic\n", "exit_code": 0},
    }
    return {
        "job_id": "job_test",
        "status": status,
        "result": results[operation] if status == "succeeded" else None,
        "error": None if status == "succeeded" else {
            "error_class": "code",
            "reason": "code_execution_failed" if status == "failed" else "cancelled",
        },
    }


@pytest.mark.anyio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    ("terminal_status", "agent_status"),
    [("succeeded", "completed"), ("failed", "failed"), ("cancelled", "cancelled")],
)
async def test_tool_waits_for_same_job_before_http_reply_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_status: str, agent_status: str,
    operation: str,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    request = _job_request(attempt, operation)

    class Client(FakeAgateClient):
        def get_job(self, job_id, wait=False, timeout=30.0, include_spec=False):
            # No intermediate result may be sealed or replayed while polling.
            assert control.get_operation_artifact(
                attempt.id, request["idempotency_key"], GatewayOperation(operation),
            ) is None
            self.fetched.append((job_id, wait, timeout))
            status = ["queued", "running", terminal_status][len(self.fetched) - 1]
            if status in {"queued", "running"}:
                return {"job_id": job_id, "status": status, "result": None}
            return self.job

    client = Client(_terminal_job(terminal_status, operation), acceptance_job_id="job_test")
    adapter, _, jobs = _adapter(tmp_path, client)
    monkeypatch.setattr(service, "_adapter", adapter)
    store_result = service._store_gateway_result

    def store_terminal_only(*args, **kwargs):
        assert len(client.fetched) == 3
        return store_result(*args, **kwargs)

    monkeypatch.setattr(service, "_store_gateway_result", store_terminal_only)
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    headers = {"Authorization": f"Bearer {capability.token}"}
    try:
        with anyio.fail_after(10):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://runtime",
            ) as http:
                response = await http.post("/v1/operations", json=request, headers=headers)
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["status"] == agent_status
                assert body["result"]["status"] == terminal_status
                if terminal_status != "succeeded":
                    assert body["result"]["error"]["reason"] == client.job["error"]["reason"]
                elif operation == "profile":
                    kernel = body["result"]["result"]["kernels"][0]
                    assert kernel["name"] == "vector_add" and kernel["duration_us"] == 4.0
                else:
                    for key, value in client.job["result"].items():
                        assert body["result"]["result"][key] == value

                # Replays and Artifact reads see only the terminal snapshot, without Agate calls.
                replay = await http.post("/v1/operations", json=request, headers=headers)
                assert replay.json() == body
                read = await http.post("/v1/runtime/queries", headers=headers, json={
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": "read-result",
                    "operation": "result_artifact_read",
                    "result_artifact_digest": body["result_artifact_digest"],
                })
                assert read.status_code == 200, read.text
                assert read.json()["result"]["status"] == agent_status
                assert read.json()["result"]["result"] == body["result"]
        assert client.fetched == [("job_test", False, 30.0)] * 3
        assert [kind for kind, _ in client.submitted] == [
            "compile" if operation == "check" else operation,
        ]
        assert client.cancelled == []
        assert len(jobs.list_owned(attempt.id)) == 1
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_interrupted_tool_wait_resumes_same_job_without_queued_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient(
        {"job_id": "job_test", "status": "running"}, acceptance_job_id="job_test",
    )
    adapter, _, jobs = _adapter(tmp_path, client)
    monkeypatch.setattr(service, "_adapter", adapter)
    request = _job_request(attempt, operation)
    payload = json.dumps(request).encode()
    try:
        with anyio.move_on_after(0.2) as scope:
            await service.execute(capability.token, payload)
        assert scope.cancel_called
        assert control.get_operation_artifact(
            attempt.id, request["idempotency_key"], GatewayOperation(operation),
        ) is None
        assert jobs.find_request(attempt.id, request["idempotency_key"]).job_id == "job_test"
        # The polling backoff yields and prevents a busy loop, even on immediate replies.
        assert client.fetched == [("job_test", False, 30.0)]

        client.job = _terminal_job("succeeded", operation)
        with anyio.fail_after(5):
            completed = await service.execute(capability.token, payload)
        assert completed.status == "completed"
        assert client.fetched == [("job_test", False, 30.0)] * 2
        assert len(client.submitted) == 1
        assert client.cancelled == []
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_proxy_refuses_to_freeze_a_nonterminal_job(tmp_path: Path, operation: str) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    request = _job_request(attempt, operation)
    payload = json.dumps(request).encode()
    adapter.result = GatewayAdapterResult(
        status="queued", result={"job_id": "job_test", "status": "running"}, job_id="job_test",
    )
    try:
        with pytest.raises(InfrastructureError, match="did not reach a terminal state"):
            await service.execute(capability.token, payload)
        assert control.get_operation_artifact(
            attempt.id, request["idempotency_key"], GatewayOperation(operation),
        ) is None
        adapter.result = GatewayAdapterResult(
            status="completed", result=_terminal_job("succeeded", operation), job_id="job_test",
        )
        assert (await service.execute(capability.token, payload)).status == "completed"
        assert len(adapter.requests) == 2
    finally:
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_unknown_job_status_is_not_waited_or_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient(
        {"job_id": "job_test", "status": "unexpected"}, acceptance_job_id="job_test",
    )
    adapter, _, jobs = _adapter(tmp_path, client)
    monkeypatch.setattr(service, "_adapter", adapter)
    request = _job_request(attempt, operation)
    try:
        with anyio.fail_after(5), pytest.raises(InfrastructureError, match="unknown job status"):
            await service.execute(capability.token, json.dumps(request).encode())
        assert control.get_operation_artifact(
            attempt.id, request["idempotency_key"], GatewayOperation(operation),
        ) is None
        assert client.fetched == [("job_test", False, 30.0)]
        assert len(client.submitted) == 1
    finally:
        jobs.close()
        control.close()
        registry.close()
