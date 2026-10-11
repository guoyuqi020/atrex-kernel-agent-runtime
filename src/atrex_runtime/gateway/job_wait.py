"""Keep synchronous Runtime calls attached to an Agate Job until it finishes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import anyio

from ..artifacts.local import JsonValue
from ..domain.errors import InfrastructureError

if TYPE_CHECKING:
    from .agate import AgateClient

AgateCall = Callable[[Callable[[], dict[str, object]]], Awaitable[dict[str, JsonValue]]]


async def wait_for_terminal_job(
    client: AgateClient,
    job_id: str,
    *,
    call: AgateCall,
    wait_timeout_s: float,
) -> dict[str, JsonValue]:
    """Renew expired long polls on the same Job, preserving transport retry policy.

    The SDK timeout bounds one long poll, not the lifetime of a Runtime operation.
    A queued/running reply must neither trigger resubmission nor become an immutable
    Result Artifact. Cancellation propagates to the caller without replacing the Job.
    """
    while True:
        job = await call(lambda: client.get_job(job_id, wait=True, timeout=wait_timeout_s))
        status = job.get("status")
        if status in ("succeeded", "failed", "cancelled"):
            return job
        if status not in ("queued", "running"):
            raise InfrastructureError("Agate returned an unknown job status")
        # Some backends return immediately despite wait=True; yield and avoid a hot loop.
        await anyio.sleep(1.0)
