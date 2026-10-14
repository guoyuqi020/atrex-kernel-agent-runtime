"""Runtime async boundary around the unchanged synchronous Agate SDK.

Only bounded SDK requests run in threads. Job polling and persistent transport
backoff belong to coroutines; neither borrows the default AnyIO worker capacity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast

import anyio
from anyio.lowlevel import RunVar

from .request_scope import check_execution

_LIMITS = {"request": 32, "upload": 8, "control": 4, "metadata": 4, "prepare": 4}
_LIMITERS: RunVar[dict[str, anyio.CapacityLimiter]] = RunVar("agate_sdk_limiters")


def sdk_limiter(lane: str) -> anyio.CapacityLimiter:
    """Separate network/packing work from session runners and control requests."""
    try:
        limiters = _LIMITERS.get()
    except LookupError:
        limiters = {}
        _LIMITERS.set(limiters)
    if lane not in limiters:
        limiters[lane] = anyio.CapacityLimiter(_LIMITS[lane])
    return limiters[lane]


async def bounded_sync[T](operation: Callable[[], T], *, lane: str = "request") -> T:
    def execute() -> T:
        # Recheck after queueing for capacity: cancellation/recovery may revoke us.
        check_execution()
        return operation()

    value = await anyio.to_thread.run_sync(execute, limiter=sdk_limiter(lane))
    await anyio.lowlevel.checkpoint()
    check_execution()
    return value


@dataclass(frozen=True)
class SdkCall:
    """A typed call descriptor, also callable by legacy synchronous test adapters."""

    client: object
    method: str
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)

    def __call__(self) -> Any:
        return getattr(self.client, self.method)(*self.args, **self.kwargs)

    async def run(self) -> Any:
        check_execution()
        # Inspect the class: wrapper __getattr__ must not skip an outer adapter.
        handler = getattr(type(self.client), "call_async", None)
        if handler is not None:
            return await handler(self.client, self.method, *self.args, **self.kwargs)
        lane = (
            "control"
            if self.method in {"health", "cancel_job"}
            else "metadata"
            if self.method in {"get_env", "get_capabilities", "list_env"}
            else "upload"
            if self.method in {"upload_file", "prepare_uploads"}
            else "request"
        )
        return await bounded_sync(self, lane=lane)


def sdk_call(client: object, method: str, *args: object, **kwargs: object) -> SdkCall:
    return SdkCall(client, method, args, kwargs)


async def run_sdk_call[T](operation: Callable[[], T]) -> T:
    if isinstance(operation, SdkCall):
        return cast(T, await operation.run())
    return await bounded_sync(operation)


async def call_method(client: object, method: str, *args: object, **kwargs: object) -> Any:
    return await sdk_call(client, method, *args, **kwargs).run()


async def prepare_call[T](operation: Callable[..., T], *args: object) -> T:
    return await bounded_sync(partial(operation, *args), lane="prepare")
