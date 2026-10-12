"""Carry an HTTP execution fence into synchronous Agate transport retries."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class GatewayExecutionSupersededError(PermissionError):
    """An old Session may no longer submit, poll, or publish Gateway work."""


_CHECK: ContextVar[Callable[[], None] | None] = ContextVar("gateway_execution_check", default=None)


def check_execution() -> None:
    """Check the current fence, including from AnyIO's propagated worker context."""
    check = _CHECK.get()
    if check is not None:
        check()


@contextmanager
def execution_scope(check: Callable[[], None]) -> Iterator[None]:
    token = _CHECK.set(check)
    try:
        check_execution()
        yield
    finally:
        _CHECK.reset(token)
