"""Uniform persistent retry policy for synchronous Agate SDK requests."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar, cast

_T = TypeVar("_T")
_EXPONENTIAL_BACKOFF_ERROR_LIMIT = 5
_INITIAL_BACKOFF_SECONDS = 5.0
_MAX_BACKOFF_SECONDS = 40.0
_STEADY_RETRY_SECONDS = 60.0
RETRYABLE_CLIENT_STATUSES = frozenset({408, 429})
_LOGGER = logging.getLogger(__name__)


def _is_permanent(error: Exception) -> bool:
    """A 4xx other than request-timeout/overload cannot succeed on retry."""
    status = getattr(error, "status", None)
    if not isinstance(status, int):
        return False
    return 400 <= status < 500 and status not in RETRYABLE_CLIENT_STATUSES


class RetryingAgateClient:
    """Apply one retry contract to every method used from the Agate SDK.

    Transient failures use delays of 5, 10, 20, and 40 seconds for the first
    five attempts. After the fifth consecutive failure, the request is retried
    every 60 seconds without a terminal attempt limit. Any successful response
    ends the sequence, so the next request starts with a fresh error count. A
    permanent 4xx other than 408 and 429 is raised immediately because repeating
    an invalid or unauthorized request cannot recover without changing it.
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
        error_count = 0
        while True:
            try:
                return request()
            except Exception as error:
                error_count += 1
                if _is_permanent(error):
                    _LOGGER.debug(
                        "Agate %s failed permanently; not retrying: %s: %s",
                        operation,
                        type(error).__name__,
                        error,
                    )
                    raise
                if error_count < _EXPONENTIAL_BACKOFF_ERROR_LIMIT:
                    delay = min(
                        _MAX_BACKOFF_SECONDS,
                        _INITIAL_BACKOFF_SECONDS * (2 ** (error_count - 1)),
                    )
                    retry_state = (
                        f"{error_count}/{_EXPONENTIAL_BACKOFF_ERROR_LIMIT}"
                    )
                else:
                    delay = _STEADY_RETRY_SECONDS
                    retry_state = f"persistent attempt {error_count}"
                _LOGGER.warning(
                    "Agate %s failed (%s); retrying in %.1f seconds: %s: %s",
                    operation,
                    retry_state,
                    delay,
                    type(error).__name__,
                    error,
                )
                self._sleeper(delay)
