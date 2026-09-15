"""Recover terminal infrastructure failures at the individual Job boundary."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable

import anyio

from ..artifacts.local import JsonValue

_LOGGER = logging.getLogger(__name__)
JobExecution = tuple[str, dict[str, JsonValue]]


def _requires_resubmission(job: dict[str, JsonValue]) -> bool:
    error = job.get("error")
    if job.get("status") != "failed" or not isinstance(error, dict):
        return False
    return error.get("error_class") == "infra"


async def run_with_job_recovery(
    payload: dict[str, object],
    execute: Callable[[dict[str, object]], Awaitable[JobExecution]],
) -> JobExecution:
    """Repeat submit/bind/collect for explicitly classified infrastructure failures.

    Transport retries remain the SDK wrapper's responsibility. A replacement key
    is stable for its failed Job, so replaying the same recovery does not create
    another replacement. Every replacement goes through normal binding/events.
    Completed sibling batches stay intact because recovery runs inside each Job.
    Candidate errors, unknown failures and cancelled Jobs are not resubmitted.
    Cancellation propagates, including during the persistent backoff.
    """
    request = dict(payload)
    failures = 0
    while True:
        job_id, job = await execute(request)
        if not _requires_resubmission(job):
            return job_id, job
        error = job["error"]
        assert isinstance(error, dict)
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
        prefix = "logs-retry:" if error.get("reason") == "logs_unavailable" else "infra-retry:"
        request = {
            **payload,
            "idempotency_key": prefix + hashlib.sha256(job_id.encode()).hexdigest(),
        }
