"""Recover orphaned Agate evaluations without losing completed shape batches."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from ..artifacts.local import JsonValue
from ..domain.errors import InfrastructureError

if TYPE_CHECKING:
    from .agate import AgateClient, AgateJobBinding, SqliteAgateJobStore
    from .job_recovery import JobExecution

AgateCall = Callable[[Callable[[], dict[str, object]]], Awaitable[dict[str, JsonValue]]]
_LOGGER = logging.getLogger(__name__)


def _missing_job(error: BaseException) -> bool:
    """Inspect the SDK error through the Runtime's public error wrapper."""
    current: BaseException | None = error
    while current is not None:
        if vars(current).get("status") in (404, 410):
            return True
        current = current.__cause__
    return False


async def execute_bound_job(
    client: AgateClient,
    jobs: SqliteAgateJobStore | None,
    binding: AgateJobBinding,
    submission: dict[str, object],
    *,
    call: AgateCall,
    submit_call: AgateCall,
    wait_timeout_s: float,
    recover: bool,
) -> JobExecution:
    """Probe a dead executor's job, reuse terminal evidence or replace stale work.

    Transport/authentication errors preserve the binding. Replacement acceptance
    precedes the atomic binding update; its deterministic key covers a crash in
    that gap. The caller holds the logical Evaluate execution lease throughout.
    """
    if recover and jobs is None:
        raise InfrastructureError("Agate recovery requires a durable job store")
    existing = (
        None if jobs is None else jobs.find_request(binding.attempt_id, binding.idempotency_key)
    )
    replacement = False
    if existing is not None:
        assert jobs is not None
        jobs.bind(replace(binding, job_id=existing.job_id))
        if recover:
            try:
                old = await call(lambda: client.get_job(existing.job_id, wait=False))
            except Exception as error:
                if not _missing_job(error):
                    raise InfrastructureError(
                        "Agate recovery could not query the original job"
                    ) from error
                old = None
            if old is not None:
                status = old.get("status")
                if status == "succeeded" and isinstance(old.get("result"), dict):
                    _LOGGER.info("Recovered Agate result for job %s", existing.job_id)
                    return existing.job_id, old
                if status not in {
                    "queued", "running", "pending", "cancelled", "failed", "succeeded",
                }:
                    raise InfrastructureError("Agate recovery returned an unknown job status")
                if status in {"queued", "running", "pending"}:
                    try:
                        await call(lambda: client.cancel_job(existing.job_id))
                    except Exception as error:
                        if not _missing_job(error):
                            raise InfrastructureError(
                                "Agate recovery could not cancel the original job"
                            ) from error
            replacement = True
            submission = {
                **submission,
                "idempotency_key": "orphan-retry:"
                + hashlib.sha256(existing.job_id.encode()).hexdigest(),
            }
    if existing is None or replacement:
        accepted = await submit_call(lambda: client.submit_job(binding.kind, submission))
        job_id = accepted.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise InfrastructureError("Agate acceptance did not contain a valid job_id")
        if jobs is not None:
            if replacement:
                assert existing is not None
                jobs.replace_job(existing, job_id)
                _LOGGER.info("Replaced orphaned Agate job %s with %s", existing.job_id, job_id)
            else:
                jobs.bind(replace(binding, job_id=job_id))
    else:
        job_id = existing.job_id
    job = await call(lambda: client.get_job(job_id, wait=True, timeout=wait_timeout_s))
    return job_id, job
