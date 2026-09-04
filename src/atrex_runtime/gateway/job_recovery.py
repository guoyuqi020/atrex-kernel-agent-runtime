"""Resubmit executions whose successful backend lost its result logs."""

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
    details = error.get("details")
    return (
        error.get("error_class") == "infra"
        and error.get("reason") == "logs_unavailable"
        and isinstance(details, dict)
        and details.get("backend_state") == "succeeded"
    )


async def run_with_log_recovery(
    payload: dict[str, object],
    execute: Callable[[dict[str, object]], Awaitable[JobExecution]],
) -> JobExecution:
    """Repeat submit/bind/collect, not get_job, for this one terminal failure.

    Transport retries remain the SDK wrapper's responsibility. A replacement key
    is stable for its failed Job, so replaying the same recovery does not create
    another replacement. Every replacement goes through normal binding/events.
    Cancellation propagates, including during the persistent backoff.
    """
    request = dict(payload)
    failures = 0
    while True:
        job_id, job = await execute(request)
        if not _requires_resubmission(job):
            return job_id, job
        failures += 1
        delay = float(5 * 2 ** (failures - 1)) if failures < 5 else 60.0
        _LOGGER.warning(
            "Agate job %s failed: infra/logs_unavailable, backend_state=succeeded; "
            "resubmitting a new job in %.1fs (failure %d)",
            job_id, delay, failures,
        )
        await anyio.sleep(delay)
        request = {
            **payload,
            "idempotency_key": "logs-retry:" + hashlib.sha256(job_id.encode()).hexdigest(),
        }
