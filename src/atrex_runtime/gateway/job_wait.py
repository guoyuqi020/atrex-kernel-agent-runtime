"""Keep synchronous Runtime calls attached to an Agate Job until it finishes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import anyio

from ..artifacts.local import JsonValue
from ..domain.errors import InfrastructureError
from .async_sdk import sdk_call

if TYPE_CHECKING:
    from .agate import AgateClient

AgateCall = Callable[[Callable[[], dict[str, object]]], Awaitable[dict[str, JsonValue]]]


async def wait_for_terminal_job(
    client: AgateClient,
    job_id: str,
    *,
    call: AgateCall,
    wait_timeout_s: float,
    include_spec: bool = False,
) -> dict[str, JsonValue]:
    """Query the same Job with bounded SDK calls and thread-free polling gaps.

    wait_timeout_s bounds a single request (capped at 30 seconds), never the
    lifetime of the tool. Only a terminal result may be published to the Agent.
    """
    delay = 1.0
    previous_status = None
    while True:
        kwargs: dict[str, object] = {"wait": False, "timeout": min(wait_timeout_s, 30.0)}
        if include_spec:
            kwargs["include_spec"] = True
        job = await call(sdk_call(client, "get_job", job_id, **kwargs))
        status = job.get("status")
        if status in ("succeeded", "failed", "cancelled"):
            return job
        if status not in ("queued", "running"):
            raise InfrastructureError("Agate returned an unknown job status")
        # Gradually reduce traffic for unchanged jobs; status changes reset the pace.
        if status != previous_status:
            delay = 1.0
        await anyio.sleep(delay)
        previous_status = status
        delay = min(10.0, delay * 2.0)
