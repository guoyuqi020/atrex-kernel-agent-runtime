"""Uniform bounded retry policy for synchronous Agate SDK requests."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar, cast

_T = TypeVar("_T")
_MAX_CONSECUTIVE_ERRORS = 5
_INITIAL_BACKOFF_SECONDS = 5.0
_MAX_BACKOFF_SECONDS = 40.0
_LOGGER = logging.getLogger(__name__)


class RetryingAgateClient:
    """Apply one retry contract to every method used from the Agate SDK.

    A request is attempted at most five times. The four delays before the
    terminal attempt are 5, 10, 20, and 40 seconds. Any successful response ends
    the sequence, so the next request starts with a fresh consecutive-error
    count.
    """

    def __init__(
        self,
        client: object,
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._sleeper = sleeper

    @property
    def wrapped_client(self) -> object:
        """Expose the SDK client for diagnostics without bypassing the policy."""
        return self._client

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        return self._call("submit_job", lambda: self._client.submit_job(kind, request))  # type: ignore[attr-defined]

    def get_job(
        self,
        job_id: str,
        wait: bool = False,
        timeout: float = 30.0,
        include_spec: bool = False,
    ) -> dict[str, object]:
        return self._call(
            "get_job",
            lambda: self._client.get_job(  # type: ignore[attr-defined]
                job_id,
                wait=wait,
                timeout=timeout,
                include_spec=include_spec,
            ),
        )

    def cancel_job(self, job_id: str) -> dict[str, object]:
        return self._call("cancel_job", lambda: self._client.cancel_job(job_id))  # type: ignore[attr-defined]

    def list_env(self, force: bool = False) -> list[dict[str, object]]:
        return self._call("list_env", lambda: self._client.list_env(force=force))  # type: ignore[attr-defined]

    def get_env(self, gpu: str, force: bool = False) -> dict[str, object]:
        return self._call("get_env", lambda: self._client.get_env(gpu, force=force))  # type: ignore[attr-defined]

    def get_capabilities(self, gpu: str, force: bool = False) -> dict[str, object]:
        return self._call(
            "get_capabilities",
            lambda: self._client.get_capabilities(gpu, force=force),  # type: ignore[attr-defined]
        )

    def health(self) -> bool:
        return cast(
            bool,
            self._call("health", lambda: self._client.health()),  # type: ignore[attr-defined]
        )

    def _call(self, operation: str, request: Callable[[], _T]) -> _T:
        for error_count in range(1, _MAX_CONSECUTIVE_ERRORS + 1):
            try:
                return request()
            except Exception as error:
                if error_count == _MAX_CONSECUTIVE_ERRORS:
                    _LOGGER.error(
                        "Agate %s failed %d consecutive times; giving up: %s: %s",
                        operation,
                        error_count,
                        type(error).__name__,
                        error,
                    )
                    raise
                delay = min(
                    _MAX_BACKOFF_SECONDS,
                    _INITIAL_BACKOFF_SECONDS * (2 ** (error_count - 1)),
                )
                _LOGGER.warning(
                    "Agate %s failed (%d/%d); retrying in %.1f seconds: %s: %s",
                    operation,
                    error_count,
                    _MAX_CONSECUTIVE_ERRORS,
                    delay,
                    type(error).__name__,
                    error,
                )
                self._sleeper(delay)
        raise AssertionError("Agate retry loop terminated without a result")
