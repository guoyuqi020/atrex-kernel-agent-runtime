"""Async Agate waits/retries must not consume the shared 40-thread capacity."""

from __future__ import annotations

from collections import Counter
from threading import Event

import anyio
import pytest
from atrex_gateway_client.errors import GatewayError
from httpx import ASGITransport, AsyncClient
from test_agate_gateway_adapter import _adapter
from test_agate_oss_client import UploadClient, _request
from test_gateway_job_wait import _job_request, _terminal_job
from test_gateway_proxy import _service

from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.agate import AgateConnectionConfig, load_agate_sdk
from atrex_runtime.gateway.async_sdk import call_method, sdk_limiter
from atrex_runtime.gateway.oss_client import OssAgateClient
from atrex_runtime.gateway.request_scope import (
    GatewayExecutionSupersededError,
    execution_scope,
)
from atrex_runtime.gateway.retrying_client import RetryingAgateClient


@pytest.mark.anyio
async def test_48_http_tools_wait_without_starving_short_calls_or_default_workers(tmp_path):
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    polled = Counter()
    accepted = []
    finish = Event()

    class Client:
        def submit_job(self, kind, request):
            job_id = f"dv_{request['command']}"
            accepted.append(job_id)
            return {"job_id": job_id}

        def get_job(self, job_id, *, wait=False, timeout=30.0):
            assert not wait and timeout <= 30
            polled[job_id] += 1
            if not finish.is_set():
                return {"job_id": job_id, "status": "running"}
            return {**_terminal_job("succeeded", "dev"), "job_id": job_id}

        def get_env(self, gpu, force=False):
            return {"gpu": gpu}

        def get_capabilities(self, gpu, force=False):
            return {"frameworks": []}

    adapter, _, jobs = _adapter(tmp_path, RetryingAgateClient(Client()))
    service._adapter = adapter
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    headers = {"Authorization": f"Bearer {capability.token}"}
    replies = []
    try:
        with anyio.fail_after(15):
            async with AsyncClient(transport=ASGITransport(app), base_url="http://runtime") as http:
                async def post(index):
                    request = _job_request(attempt, "dev")
                    request.update(idempotency_key=f"wait-{index}", command=f"job-{index}")
                    reply = await http.post("/v1/operations", json=request, headers=headers)
                    assert reply.status_code == 200, reply.text
                    assert reply.json()["result"]["status"] == "succeeded"
                    replies.append(reply)

                async with anyio.create_task_group() as group:
                    for i in range(48):
                        group.start_soon(post, i)
                    while len(polled) < 48:
                        await anyio.sleep(0.01)
                    assert not replies  # Agent HTTP requests stay pending until terminal.
                    assert anyio.to_thread.current_default_thread_limiter().borrowed_tokens == 0
                    with anyio.fail_after(1):
                        assert await anyio.to_thread.run_sync(lambda: "free") == "free"
                        env = await http.post("/v1/operations", headers=headers, json={
                            "schema_version": 2, "attempt_id": attempt.id,
                            "operation": "env", "idempotency_key": "metadata-during-wait",
                        })
                        assert env.status_code == 200, env.text
                    finish.set()
        assert len(replies) == len(accepted) == len(set(accepted)) == 48
        assert len(jobs.list_owned(attempt.id)) == 48
        assert not control._connection.execute("SELECT * FROM gateway_active_calls").fetchall()
    finally:
        finish.set()
        jobs.close()
        control.close()
        registry.close()


@pytest.mark.anyio
async def test_async_backoff_releases_all_slots_and_cancellation_stops_retry():
    entered = anyio.Event()
    retries = 0
    calls = Counter()
    delay_values = []

    class Client:
        def get_env(self, gpu):
            calls[gpu] += 1
            raise ConnectionError("temporary outage")

        def cancel_job(self, job_id):
            return {"status": "cancelled", "job_id": job_id}

    async def backoff(delay):
        nonlocal retries
        retries += 1
        delay_values.append(delay)
        if retries == 48:
            entered.set()
        await anyio.sleep_forever()

    client = RetryingAgateClient(Client(), async_sleeper=backoff)
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            for i in range(48):
                group.start_soon(call_method, client, "get_env", str(i))
            await entered.wait()
            for lane in ("control", "metadata", "request", "upload"):
                assert sdk_limiter(lane).borrowed_tokens == 0
            assert anyio.to_thread.current_default_thread_limiter().borrowed_tokens == 0
            assert (await call_method(client, "cancel_job", "dv_existing"))["status"] == "cancelled"
            group.cancel_scope.cancel()
    assert len(calls) == 48 and set(calls.values()) == {1}
    assert delay_values == [5.0] * 48


@pytest.mark.anyio
async def test_cancel_can_run_while_all_request_slots_are_in_use():
    started = anyio.Event()
    release = Event()
    count = 0

    def notify():
        nonlocal count
        count += 1
        if count == 32:
            started.set()

    class Client:
        def get_job(self, job_id):
            anyio.from_thread.run_sync(notify)
            assert release.wait(5)
            return {"job_id": job_id}

        def cancel_job(self, job_id):
            return {"status": "cancelled"}

    raw = Client()
    try:
        with anyio.fail_after(8):
            async with anyio.create_task_group() as group:
                for i in range(32):
                    group.start_soon(call_method, raw, "get_job", str(i))
                await started.wait()
                try:
                    with anyio.fail_after(1):
                        assert await call_method(raw, "cancel_job", "dv") == {"status": "cancelled"}
                finally:
                    release.set()
    finally:
        release.set()


@pytest.mark.anyio
async def test_async_oss_retries_keep_reservation_files_and_submit_key():
    raw = UploadClient()
    raw.failures = {"prepare": [503], "upload": [503], "submit": [503, 503]}
    delays = []

    async def backoff(delay):
        delays.append(delay)
        assert sdk_limiter("upload").borrowed_tokens == 0
        assert sdk_limiter("request").borrowed_tokens == 0
        await anyio.lowlevel.checkpoint()

    client = OssAgateClient(RetryingAgateClient(raw, async_sleeper=backoff), max_inline_bytes=1)
    assert (await call_method(client, "submit_job", "dev", _request()))["job_id"] == "dv_accepted"
    assert delays == [5, 5, 5, 10]
    assert len(raw.prepared) == 2 and len(raw.upload_paths) == 2
    assert len(raw.uploads) == 1 and len(raw.submitted) == 3
    assert len({request["idempotency_key"] for _, request in raw.submitted}) == 1
    assert all(request is raw.submitted[0][1] for _, request in raw.submitted)
    assert all(not path.exists() for path in raw.upload_paths)


@pytest.mark.anyio
async def test_async_upload_cancel_cleans_archive_without_submission():
    raw = UploadClient()
    raw.failures = {"upload": [503]}
    entered = anyio.Event()

    async def backoff(_):
        entered.set()
        await anyio.sleep_forever()

    client = OssAgateClient(RetryingAgateClient(raw, async_sleeper=backoff), max_inline_bytes=1)
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(call_method, client, "submit_job", "dev", _request())
            await entered.wait()
            assert raw.upload_paths and all(p.exists() for p in raw.upload_paths)
            group.cancel_scope.cancel()
    assert not raw.submitted
    assert all(not p.exists() for p in raw.upload_paths)


@pytest.mark.anyio
async def test_async_retry_rechecks_recovery_fence():
    revoked = False
    calls = []

    def check():
        if revoked:
            raise GatewayExecutionSupersededError("old generation")

    class Client:
        def submit_job(self, kind, request):
            calls.append(request)
            raise ConnectionError("lost acceptance")

    async def backoff(_):
        nonlocal revoked
        revoked = True

    client = RetryingAgateClient(Client(), async_sleeper=backoff)
    with execution_scope(check), pytest.raises(GatewayExecutionSupersededError):
        await call_method(client, "submit_job", "dev", {})
    assert len(calls) == 1


@pytest.mark.anyio
async def test_runtime_sdk_disables_sync_overload_retry_and_preserves_idempotency(monkeypatch):
    client, _ = load_agate_sdk(AgateConnectionConfig(
        auth_mode="none", http_timeout_s=5, wait_timeout_s=1200,
    ))
    retrying = client._client
    raw = retrying.wrapped_client
    assert raw.transport_retries == 0
    attempts = []
    delays = []

    def request(method, path, body, timeout):
        attempts.append((method, path, body))
        if len(attempts) == 1:
            raise GatewayError(429, {"reason": "gateway_overloaded"}, retry_after_s=12)
        return {}, {"job_id": "ev_existing"}

    async def backoff(delay):
        delays.append(delay)
        assert sdk_limiter("request").borrowed_tokens == 0

    def forbidden_sleep(_):
        pytest.fail("SDK must not sleep inside a worker thread")

    monkeypatch.setattr(raw, "_request", request)
    monkeypatch.setattr("atrex_gateway_client.client.time.sleep", forbidden_sleep)
    retrying._async_sleeper = backoff
    result = await call_method(client, "submit_job", "eval", {"candidate": "code"})
    assert result == {"job_id": "ev_existing"}
    assert len(attempts) == 2
    assert attempts[0][2]["idempotency_key"] == attempts[1][2]["idempotency_key"]
    assert delays == [12]


@pytest.mark.anyio
async def test_async_retries_keep_exponential_and_persistent_backoff():
    requests = []
    delays = []

    class Client:
        def submit_job(self, kind, request):
            requests.append(request)
            if len(requests) <= 7:
                raise GatewayError(503, {"reason": "temporary_outage"})
            return {"job_id": "dv_recovered"}

    async def backoff(delay):
        delays.append(delay)
        assert sdk_limiter("request").borrowed_tokens == 0

    client = RetryingAgateClient(Client(), async_sleeper=backoff)
    original = {"command": "true"}
    result = await call_method(client, "submit_job", "dev", original)
    assert result == {"job_id": "dv_recovered"}
    assert delays == [5, 10, 20, 40, 60, 60, 60]
    assert len({p["idempotency_key"] for p in requests}) == 1
    assert "idempotency_key" not in original


@pytest.mark.anyio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_async_permanent_errors_propagate_without_backoff(status):
    error = GatewayError(status, {"reason": "source_archive_invalid", "error_class": "validation"})

    class Client:
        def submit_job(self, kind, request):
            raise error

    async def forbidden_backoff(_):
        pytest.fail("candidate/authentication errors must not retry")

    client = RetryingAgateClient(Client(), async_sleeper=forbidden_backoff)
    with pytest.raises(GatewayError) as caught:
        await call_method(client, "submit_job", "eval", {})
    assert caught.value is error
