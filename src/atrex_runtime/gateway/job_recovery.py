"""Recover terminal infrastructure failures at the individual Job boundary."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable

import anyio

from ..artifacts.local import JsonValue
from ..domain.errors import InfrastructureError
from .evaluation_failures import embedded_device_failure

_LOGGER = logging.getLogger(__name__)
JobExecution = tuple[str, dict[str, JsonValue]]
EvaluationFailureRecorder = Callable[[str, dict[str, JsonValue], str, int], None]


class EvaluationJobRetriesExhausted(InfrastructureError):
    """An inferred Eval device failure exhausted its bounded replacement budget."""


def _requires_resubmission(job: dict[str, JsonValue]) -> bool:
    error = job.get("error")
    if job.get("status") != "failed" or not isinstance(error, dict):
        return False
    return error.get("error_class") == "infra"


async def run_with_job_recovery(
    payload: dict[str, object],
    execute: Callable[[dict[str, object]], Awaitable[JobExecution]],
    *,
    max_evaluation_retries: int | None = None,
    on_evaluation_failure: EvaluationFailureRecorder | None = None,
) -> JobExecution:
    """Repeat submit/bind/collect for explicitly classified infrastructure failures.

    Transport retries remain the SDK wrapper's responsibility. A replacement key
    is stable for its failed Job, so replaying the same recovery does not create
    another replacement. Every replacement goes through normal binding/events.
    Completed sibling batches stay intact because recovery runs inside each Job.
    Candidate errors, unknown failures and cancelled Jobs are not resubmitted.
    Cancellation propagates, including during the persistent backoff.

    Authoritative callers may additionally opt in to bounded recovery of device
    failures buried inside Eval/ABBA results. Exhaustion is infrastructure failure,
    not negative Kernel evidence. Their recorder seals every rejected Job first.
    """
    if max_evaluation_retries is not None and max_evaluation_retries < 0:
        raise ValueError("evaluation retry budget cannot be negative")
    request = dict(payload)
    failures = 0
    evaluation_failures = 0
    while True:
        job_id, job = await execute(request)
        explicit_infra = _requires_resubmission(job)
        diagnostic = (
            embedded_device_failure(job)
            if not explicit_infra and max_evaluation_retries is not None
            else None
        )
        if not explicit_infra and diagnostic is None:
            return job_id, job
        if diagnostic is not None:
            assert max_evaluation_retries is not None
            evaluation_failures += 1
            if on_evaluation_failure is not None:
                on_evaluation_failure(job_id, job, diagnostic, evaluation_failures)
            if evaluation_failures > max_evaluation_retries:
                raise EvaluationJobRetriesExhausted(
                    f"Agate Eval device failure after {evaluation_failures} executions: "
                    f"job_id={job_id}; {diagnostic}",
                    public_detail=f"Agate evaluation device unavailable in job {job_id}; "
                    "automatic retries exhausted. No Kernel correctness verdict was recorded.",
                )
            error: dict[str, JsonValue] = {"reason": "evaluation_device_unavailable"}
        else:
            raw_error = job["error"]
            assert isinstance(raw_error, dict)
            error = raw_error
        failures += 1
        delay = float(5 * 2 ** (failures - 1)) if failures < 5 else 60.0
        _LOGGER.warning(
            "Agate job %s failed: infra/%s (trace_id=%s); "
            "resubmitting a new job in %.1fs (failure %d)",
            job_id, error.get("reason"), error.get("trace_id", job.get("trace_id")),
            delay, failures,
        )
        await anyio.sleep(delay)
        # Keep lost-log replacement keys stable when replaying an existing recovery.
        prefix = (
            "eval-device-retry:" if diagnostic is not None
            else "logs-retry:" if error.get("reason") == "logs_unavailable"
            else "infra-retry:"
        )
        request = {
            **payload,
            "idempotency_key": prefix + hashlib.sha256(job_id.encode()).hexdigest(),
        }
