"""Profile long-poll expiry must not publish an unfinished Agent Result Artifact."""

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


def _profile_request(attempt) -> dict:
    request = json.loads(_request(attempt))
    request.update(operation="profile", idempotency_key="profile-wait", level="sol")
    request.pop("latency_prediction")
    return request


def _terminal_job(status: str) -> dict:
    return {
        "job_id": "pf_test",
        "status": status,
        "result": {"kernels": [{"name": "vector_add", "duration_us": 4.0}]}
        if status == "succeeded" else None,
        "error": None if status == "succeeded" else {
            "error_class": "code",
            "reason": "code_execution_failed" if status == "failed" else "cancelled",
        },
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("terminal_status", "agent_status"),
    [("succeeded", "completed"), ("failed", "failed"), ("cancelled", "cancelled")],
)
async def test_profile_waits_for_same_job_before_http_reply_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_status: str, agent_status: str,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    request = _profile_request(attempt)

    class Client(FakeAgateClient):
        def get_job(self, job_id, wait=False, timeout=30.0, include_spec=False):
            # No intermediate result may be sealed or replayed while polling.
            assert control.get_operation_artifact(
                attempt.id, request["idempotency_key"], GatewayOperation.PROFILE,
            ) is None
            self.fetched.append((job_id, wait, timeout))
            status = ["queued", "running", terminal_status][len(self.fetched) - 1]
            if status in {"queued", "running"}:
                return {"job_id": job_id, "status": status, "result": None}
            return self.job

    client = Client(_terminal_job(terminal_status), acceptance_job_id="pf_test")
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

                # Replays and Artifact reads see only the terminal snapshot, without Agate calls.
                replay = await http.post("/v1/operations", json=request, headers=headers)
                assert replay.json() == body
                read = await http.post("/v1/runtime/queries", headers=headers, json={
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": "read-profile",
                    "operation": "result_artifact_read",
                    "result_artifact_digest": body["result_artifact_digest"],
                })
                assert read.status_code == 200, read.text
                assert read.json()["result"]["status"] == agent_status
                assert read.json()["result"]["result"] == body["result"]
        assert client.fetched == [("pf_test", True, 1200)] * 3
        assert [kind for kind, _ in client.submitted] == ["profile"]
        assert client.cancelled == []
        assert len(jobs.list_owned(attempt.id)) == 1
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_interrupted_profile_wait_resumes_same_job_without_queued_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient(
        {"job_id": "pf_test", "status": "running"}, acceptance_job_id="pf_test",
    )
    adapter, _, jobs = _adapter(tmp_path, client)
    monkeypatch.setattr(service, "_adapter", adapter)
    request = _profile_request(attempt)
    payload = json.dumps(request).encode()
    try:
        with anyio.move_on_after(0.2) as scope:
            await service.execute(capability.token, payload)
        assert scope.cancel_called
        assert control.get_operation_artifact(
            attempt.id, request["idempotency_key"], GatewayOperation.PROFILE,
        ) is None
        assert jobs.find_request(attempt.id, request["idempotency_key"]).job_id == "pf_test"
        # The polling backoff yields and prevents a busy loop, even on immediate replies.
        assert client.fetched == [("pf_test", True, 1200)]

        client.job = _terminal_job("succeeded")
        with anyio.fail_after(5):
            completed = await service.execute(capability.token, payload)
        assert completed.status == "completed"
        assert client.fetched == [("pf_test", True, 1200)] * 2
        assert len(client.submitted) == 1
        assert client.cancelled == []
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_proxy_refuses_to_freeze_a_nonterminal_profile(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    request = _profile_request(attempt)
    payload = json.dumps(request).encode()
    adapter.result = GatewayAdapterResult(
        status="queued", result={"job_id": "pf_test", "status": "running"}, job_id="pf_test",
    )
    try:
        with pytest.raises(InfrastructureError, match="Profile did not reach a terminal state"):
            await service.execute(capability.token, payload)
        assert control.get_operation_artifact(
            attempt.id, request["idempotency_key"], GatewayOperation.PROFILE,
        ) is None
        adapter.result = GatewayAdapterResult(
            status="completed", result=_terminal_job("succeeded"), job_id="pf_test",
        )
        assert (await service.execute(capability.token, payload)).status == "completed"
        assert len(adapter.requests) == 2
    finally:
        control.close()
        registry.close()
