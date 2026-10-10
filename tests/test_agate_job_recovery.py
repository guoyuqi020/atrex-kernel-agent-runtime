"""Terminal infrastructure failures require new executions, not repeated polling."""

from __future__ import annotations

from copy import deepcopy

import anyio
import pytest

from atrex_runtime.artifacts.local import JsonValue
from atrex_runtime.gateway.job_recovery import JobExecution, run_with_job_recovery
from atrex_runtime.gateway.retrying_client import RetryingAgateClient


def lost_logs() -> dict[str, JsonValue]:
    return {
        "status": "failed",
        "execution_phase": "terminal",
        "result": None,
        "error": {
            "error_class": "infra",
            "reason": "logs_unavailable",
            "details": {"backend_state": "succeeded", "gc_deferred": True},
        },
    }


@pytest.mark.anyio
@pytest.mark.parametrize("with_key", [False, True])
@pytest.mark.parametrize("error", [
    lost_logs()["error"],
    {"error_class": "infra", "reason": "exec_failed",
     "message": "runtime_env setup failed", "trace_id": "req-ray-env"},
    {"error_class": "infra", "reason": "deps_install_failed", "details": None},
    {"error_class": "infra", "reason": "logs_unavailable",
     "details": {"backend_state": "failed"}},
    {"error_class": "infra"},
])
async def test_resubmits_with_fresh_keys_until_success(
    monkeypatch: pytest.MonkeyPatch, with_key: bool, error: dict[str, JsonValue],
) -> None:
    payload: dict[str, object] = {"files": {"kernel.py": "source"}, "lock_clocks": True}
    if with_key:
        payload["idempotency_key"] = "original"
    original = deepcopy(payload)
    requests: list[dict[str, object]] = []
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    async def execute(request: dict[str, object]) -> JobExecution:
        requests.append(request)
        job_id = f"ev_{len(requests)}"
        return job_id, {"job_id": job_id, **(
            {"status": "failed", "error": error} if len(requests) < 8
            else {"status": "succeeded"}
        )}

    monkeypatch.setattr("atrex_runtime.gateway.job_recovery.anyio.sleep", sleep)
    job_id, job = await run_with_job_recovery(payload, execute)
    assert job_id == "ev_8" and job["status"] == "succeeded"
    assert delays == [5, 10, 20, 40, 60, 60, 60]
    assert len({r["idempotency_key"] for r in requests[1:]}) == 7
    prefix = "logs-retry:" if error.get("reason") == "logs_unavailable" else "infra-retry:"
    assert all(str(r["idempotency_key"]).startswith(prefix) for r in requests[1:])
    assert all({k: v for k, v in r.items() if k != "idempotency_key"} ==
               {k: v for k, v in payload.items() if k != "idempotency_key"} for r in requests)
    assert payload == original
    assert requests[0] == original

    # A restarted execution seeing the same failed Job uses the same replacement key.
    first_replacement_key = requests[1]["idempotency_key"]
    requests.clear()
    await run_with_job_recovery(payload, execute)
    assert requests[1]["idempotency_key"] == first_replacement_key


@pytest.mark.anyio
@pytest.mark.parametrize("patch", [
    {"status": "succeeded"}, {"status": "cancelled"}, {"status": "running"},
    {"error": None}, {"error": "logs_unavailable"},
    {"error": {"error_class": "candidate", "reason": "logs_unavailable",
               "details": {"backend_state": "succeeded"}}},
    {"error": {"error_class": "compile", "reason": "exec_failed"}},
    {"error": {"error_class": "correctness", "reason": "mismatch"}},
    {"error": {"reason": "exec_failed"}},
    {"error": {"error_class": "unknown", "reason": "exec_failed"}},
])
async def test_other_job_results_are_not_retried(patch: dict[str, JsonValue]) -> None:
    job = {**lost_logs(), **patch}
    calls = 0

    async def execute(_request: dict[str, object]) -> JobExecution:
        nonlocal calls
        calls += 1
        assert calls == 1
        return "ev_original", job

    assert await run_with_job_recovery({}, execute) == ("ev_original", job)


@pytest.mark.anyio
async def test_exceptions_keep_existing_handling() -> None:
    async def execute(_request: dict[str, object]) -> JobExecution:
        raise ValueError("permanent error")

    with pytest.raises(ValueError, match="permanent error"):
        await run_with_job_recovery({}, execute)


@pytest.mark.anyio
@pytest.mark.parametrize("reason", ["logs_unavailable", "exec_failed"])
async def test_cancel_during_backoff_does_not_submit_again(reason: str) -> None:
    calls = 0
    with anyio.CancelScope() as scope:
        async def execute(_request: dict[str, object]) -> JobExecution:
            nonlocal calls
            calls += 1
            scope.cancel()
            return "ev_failed", {"status": "failed", "error": {
                "error_class": "infra", "reason": reason,
            }}

        await run_with_job_recovery({}, execute)
    assert scope.cancelled_caught and calls == 1


@pytest.mark.anyio
async def test_cancel_during_persistent_backoff_stops_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    delays: list[float] = []
    with anyio.CancelScope() as scope:
        async def sleep(delay: float) -> None:
            delays.append(delay)
            if delay == 60:
                scope.cancel()
            await anyio.lowlevel.checkpoint()

        async def execute(_request: dict[str, object]) -> JobExecution:
            nonlocal calls
            calls += 1
            return f"ev_{calls}", {"status": "failed", "error": {
                "error_class": "infra", "reason": "exec_failed",
            }}

        monkeypatch.setattr("atrex_runtime.gateway.job_recovery.anyio.sleep", sleep)
        await run_with_job_recovery({}, execute)
    assert scope.cancelled_caught and calls == 5
    assert delays == [5, 10, 20, 40, 60]


def test_explicit_poll_does_not_retry_or_resubmit() -> None:
    class Client:
        def get_job(self, *_args: object, **_kwargs: object) -> dict[str, JsonValue]:
            return lost_logs()

    def unexpected_sleep(_delay: float) -> None:
        pytest.fail("A read-only poll must not own execution recovery")

    result = RetryingAgateClient(Client(), sleeper=unexpected_sleep).get_job("ev_failed")
    assert result == lost_logs()


@pytest.mark.anyio
async def test_embedded_device_retry_limit_and_original_payload_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation_failure_fixture import DEVICE_ERROR, device_failure_result

    from atrex_runtime.gateway.job_recovery import EvaluationJobRetriesExhausted

    payload = {"idempotency_key": "original", "candidate": "source", "lock_clocks": True}
    requests: list[dict[str, object]] = []
    failures: list[tuple[str, dict[str, JsonValue], str, int]] = []
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    async def execute(request: dict[str, object]) -> JobExecution:
        requests.append(request)
        job_id = f"ev_{len(requests)}"
        return job_id, {
            "job_id": job_id, "status": "succeeded", "result": device_failure_result("10"),
        }

    monkeypatch.setattr("atrex_runtime.gateway.job_recovery.anyio.sleep", sleep)
    with pytest.raises(EvaluationJobRetriesExhausted, match="after 3 executions"):
        await run_with_job_recovery(
            payload, execute, max_evaluation_retries=2,
            on_evaluation_failure=lambda *args: failures.append(args),
        )
    assert len(requests) == len(failures) == 3
    assert delays == [5, 10]
    assert all(f[2] == DEVICE_ERROR for f in failures)
    assert len({r["idempotency_key"] for r in requests}) == 3
    assert all(r["candidate"] == "source" and r["lock_clocks"] is True for r in requests)
    assert payload["idempotency_key"] == "original"
    replacement_key = requests[1]["idempotency_key"]
    requests.clear()
    with pytest.raises(EvaluationJobRetriesExhausted):
        await run_with_job_recovery(payload, execute, max_evaluation_retries=2)
    assert requests[1]["idempotency_key"] == replacement_key


@pytest.mark.anyio
async def test_embedded_device_retry_backoff_remains_cancellable() -> None:
    from evaluation_failure_fixture import device_failure_result

    calls = 0
    with anyio.CancelScope() as scope:
        async def execute(_request: dict[str, object]) -> JobExecution:
            nonlocal calls
            calls += 1
            scope.cancel()
            return "ev_cancel", {"status": "succeeded", "result": device_failure_result("10")}

        await run_with_job_recovery({}, execute, max_evaluation_retries=2)
    assert scope.cancelled_caught and calls == 1
