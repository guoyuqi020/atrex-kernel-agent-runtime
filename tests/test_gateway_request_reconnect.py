"""Job submission, reconnects, and generation fences through the real HTTP boundary."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from copy import copy
from datetime import timedelta

import anyio
import pytest
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import FakeAgateClient, _adapter
from test_gateway_job_wait import OPERATIONS, _job_request, _terminal_job
from test_gateway_proxy import NOW_DATETIME, _service

from atrex_runtime.domain.errors import InfrastructureError
from atrex_runtime.gateway import (
    GatewayCapabilityPolicy,
    GatewayOperation,
    GatewayProxyAsgiApp,
    GatewayProxyLimits,
    SqliteGatewayControl,
)
from atrex_runtime.gateway.retrying_client import RetryingAgateClient


@pytest.mark.anyio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("owner_fails", [False, True])
async def test_reconnects_serialize_jobs_across_control_instances(
    tmp_path, monkeypatch, operation, owner_fails
):
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient(_terminal_job("succeeded", operation), acceptance_job_id="job_test")
    adapter, _, jobs = _adapter(tmp_path, client)
    service._adapter = adapter
    second = SqliteGatewayControl(
        tmp_path / "gateway.sqlite", registry, signing_key=b"p" * 32, clock=lambda: NOW_DATETIME
    )
    reconnect = copy(service)
    reconnect._control = second
    started, release, waiting = anyio.Event(), anyio.Event(), anyio.Event()
    original_execute = adapter.execute
    calls = []

    async def delayed(request):
        calls.append(request)
        result = await original_execute(request)
        if len(calls) == 1:
            started.set()
            await release.wait()
            if owner_fails:
                raise InfrastructureError("executor exited after binding the Agate job")
        return result

    monkeypatch.setattr(adapter, "execute", delayed)
    original_lease = second.request_execution

    @contextmanager
    def observed(authorization):
        with original_lease(authorization) as acquired:
            if not acquired:
                waiting.set()
            yield acquired

    monkeypatch.setattr(second, "request_execution", observed)
    responses = {}

    async def post(name, target):
        app = GatewayProxyAsgiApp(target, GatewayProxyLimits(65536, 8, 16384))
        async with AsyncClient(transport=ASGITransport(app), base_url="http://runtime") as http:
            responses[name] = await http.post(
                "/v1/operations",
                json=_job_request(attempt, operation),
                headers={"Authorization": f"Bearer {capability.token}"},
            )

    try:
        with anyio.fail_after(8):
            async with anyio.create_task_group() as group:
                group.start_soon(post, "owner", service)
                await started.wait()
                group.start_soon(post, "reconnect", reconnect)
                await waiting.wait()
                assert len(calls) == len(client.submitted) == 1
                assert not responses
                with sqlite3.connect(tmp_path / "gateway.sqlite") as db:
                    assert (
                        db.execute("SELECT count(*) FROM gateway_active_calls").fetchone()[0] == 1
                    )
                release.set()
        assert responses["owner"].status_code == (503 if owner_fails else 200)
        assert responses["reconnect"].status_code == 200, responses["reconnect"].text
        assert responses["reconnect"].json()["status"] == "completed"
        if not owner_fails:
            assert responses["owner"].json() == responses["reconnect"].json()
        assert len(client.submitted) == 1
        assert not client.cancelled
        assert len(jobs.list_owned(attempt.id)) == 1
        with sqlite3.connect(tmp_path / "gateway.sqlite") as db:
            assert db.execute("SELECT count(*) FROM gateway_active_calls").fetchone()[0] == 0
            assert db.execute("SELECT used_calls FROM gateway_capabilities").fetchone()[0] == (
                0 if operation == "dev" else 1
            )
    finally:
        second.close()
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("legacy_queued", [False, True])
async def test_recover_partial_or_legacy_nonterminal_result(
    tmp_path, monkeypatch, operation, legacy_queued
):
    from atrex_runtime.artifacts.local import ArtifactKind

    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient(_terminal_job("succeeded", operation), acceptance_job_id="job_test")
    adapter, _, jobs = _adapter(tmp_path, client)
    service._adapter = adapter
    payload = json.dumps(_job_request(attempt, operation)).encode()
    original_store = service._store_result_artifact

    def crash(**kwargs):
        raise InfrastructureError("crash between private binding and public response")

    monkeypatch.setattr(service, "_store_result_artifact", crash)
    try:
        with pytest.raises(InfrastructureError, match="between private"):
            await service.execute(capability.token, payload)
        monkeypatch.setattr(service, "_store_result_artifact", original_store)
        with sqlite3.connect(tmp_path / "gateway.sqlite") as db:
            previous = db.execute(
                "SELECT gateway_result_digest FROM gateway_operations"
            ).fetchone()[0]
        if legacy_queued:
            # Reproduce the old version's queued checkpoint (its public response
            # may already have been published too). The physical job stays bound.
            pending = service._artifacts.put_json(
                {"job_id": "job_test", "status": "queued", "result": None},
                ArtifactKind.GATEWAY_RESULT,
            )
            with sqlite3.connect(tmp_path / "gateway.sqlite") as db:
                db.execute(
                    "UPDATE gateway_operations "
                    "SET gateway_result_digest=?, result_artifact_digest=?",
                    (str(pending), str(pending)),
                )
        else:
            # A second poll would change the immutable result and previously conflicted.
            client.job["updated_at"] = "changed-after-completion"
        response = await service.execute(capability.token, payload)
        assert response.status == "completed"
        assert len(client.submitted) == 1
        assert len(client.fetched) == (2 if legacy_queued else 1)
        assert await service.execute(capability.token, payload) == response
        with sqlite3.connect(tmp_path / "gateway.sqlite") as db:
            digest = db.execute("SELECT gateway_result_digest FROM gateway_operations").fetchone()[
                0
            ]
        assert digest == previous
        service._artifacts.verify(previous)  # History was not deleted or rewritten.
        if legacy_queued:
            service._artifacts.verify(pending)
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_acceptance_loss_reuses_wire_key_after_request_restart(tmp_path, operation):
    registry, control, attempt, capability, service, _ = _service(tmp_path)

    class Client(FakeAgateClient):
        def __init__(self):
            super().__init__(_terminal_job("succeeded", operation), acceptance_job_id="job_test")
            self.keys = []

        def submit_job(self, kind, payload):
            self.keys.append(payload["idempotency_key"])
            if len(self.keys) == 1:
                accepted = super().submit_job(kind, payload)
                self.accepted = accepted
                raise ConnectionError("accepted but reply lost")
            assert self.keys[-1] == self.keys[0]
            return self.accepted

    client = Client()
    adapter, _, jobs = _adapter(tmp_path, client)
    service._adapter = adapter
    payload = json.dumps(_job_request(attempt, operation)).encode()
    try:
        with pytest.raises(InfrastructureError, match="reply lost"):
            await service.execute(capability.token, payload)
        assert not jobs.list_owned(attempt.id)
        assert (await service.execute(capability.token, payload)).status == "completed"
        assert len(client.submitted) == 1
        assert len(client.keys) == 2
        assert len(jobs.list_owned(attempt.id)) == 1
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_late_old_generation_reply_is_fenced_before_result_storage(
    tmp_path, monkeypatch, operation
):
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    client = FakeAgateClient(_terminal_job("succeeded", operation), acceptance_job_id="job_test")
    adapter, _, jobs = _adapter(tmp_path, client)
    service._adapter = adapter
    original = adapter.execute

    async def late(request):
        result = await original(request)
        registry.record_infrastructure_failure(attempt.id, "worker timeout during RPC")
        registry.retry_attempt(attempt.id)
        control.issue(
            attempt.id,
            GatewayCapabilityPolicy(
                frozenset(GatewayOperation),
                4,
                NOW_DATETIME + timedelta(hours=1),
            ),
        )
        return result

    monkeypatch.setattr(adapter, "execute", late)
    try:
        with pytest.raises(PermissionError, match="execution authority ended"):
            await service.execute(
                capability.token, json.dumps(_job_request(attempt, operation)).encode()
            )
        with sqlite3.connect(tmp_path / "gateway.sqlite") as db:
            row = db.execute(
                "SELECT gateway_result_digest,result_artifact_digest FROM gateway_operations"
            ).fetchone()
            assert row == (None, None)
            assert db.execute("SELECT count(*) FROM gateway_active_calls").fetchone()[0] == 0
        # Physical evidence stays owned by the original generation; no replacement job.
        assert len(client.submitted) == len(jobs.list_owned(attempt.id)) == 1
    finally:
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_transport_retry_stops_after_generation_changes_in_sdk_thread(tmp_path):
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    request = _job_request(attempt, "dev")

    # Use the real service so ContextVar propagation and SQLite thread ownership are covered.
    class Client(FakeAgateClient):
        def submit_job(self, kind, payload):
            self.submitted.append((kind, payload))
            anyio.from_thread.run_sync(control.revoke, attempt.id)
            raise ConnectionError("response lost just as the old session was revoked")

    raw = Client(_terminal_job("succeeded", "dev"))
    client = RetryingAgateClient(raw, sleeper=lambda _: None)
    adapter, _, jobs = _adapter(tmp_path, client)
    service._adapter = adapter
    try:
        with anyio.fail_after(5), pytest.raises(PermissionError, match="authority ended"):
            await service.execute(capability.token, json.dumps(request).encode())
        assert len(raw.submitted) == 1
        assert not jobs.list_owned(attempt.id)
    finally:
        jobs.close()
        control.close()
        registry.close()
