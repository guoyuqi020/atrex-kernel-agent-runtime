"""Uniform persistent retry policy for synchronous Agate SDK requests."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, TypeVar, cast
from uuid import uuid4

import anyio

from .async_sdk import call_method
from .request_errors import is_infrastructure_request_error, is_invalid_url_error
from .request_scope import GatewayExecutionSupersededError, check_execution

_T = TypeVar("_T")
_EXPONENTIAL_BACKOFF_ERROR_LIMIT = 5
_INITIAL_BACKOFF_SECONDS = 5.0
_MAX_BACKOFF_SECONDS = 40.0
_STEADY_RETRY_SECONDS = 60.0
RETRYABLE_CLIENT_STATUSES = frozenset({408, 429})
_LOGGER = logging.getLogger(__name__)


def _is_permanent(error: Exception) -> bool:
    """Known infrastructure reasons override an upstream 4xx classification."""
    if isinstance(error, GatewayExecutionSupersededError) or is_invalid_url_error(error):
        return True
    if is_infrastructure_request_error(error):
        return False
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
    permanent 4xx other than 408 and 429 is raised immediately. Explicit infra
    errors and oss_unavailable remain retryable even with HTTP 400/validation;
    invalid candidate sources and unauthorized requests still fail immediately.
    InvalidURL also fails immediately, including SDK envelopes mislabeled infra.
    """

    def __init__(
        self,
        client: object,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        async_sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._client = client
        self._sleeper = sleeper
        self._async_sleeper = async_sleeper

    async def call_async(self, method: str, *args: object, **kwargs: object) -> Any:
        """Retry each individual SDK stage without holding a worker during backoff."""
        if method == "submit_job":
            kind, raw = args
            request = cast(dict[str, object], raw)
            if not request.get("idempotency_key"):
                request = {**request, "idempotency_key": uuid4().hex}
            args = (kind, request)
        error_count = 0
        while True:
            check_execution()
            try:
                return await call_method(self._client, method, *args, **kwargs)
            except Exception as error:
                # Execution fences are not transient transport failures.
                check_execution()
                error_count += 1
                if _is_permanent(error):
                    raise
                delay = (
                    min(_MAX_BACKOFF_SECONDS, _INITIAL_BACKOFF_SECONDS * 2 ** (error_count - 1))
                    if error_count < _EXPONENTIAL_BACKOFF_ERROR_LIMIT
                    else _STEADY_RETRY_SECONDS
                )
                retry_after = getattr(error, "retry_after_s", None)
                details = getattr(error, "details", None)
                if retry_after is None and isinstance(details, dict):
                    with suppress(TypeError, ValueError):
                        retry_after = float(details.get("retry_after_s", 0))
                if isinstance(retry_after, (int, float)) and 0 < retry_after < float("inf"):
                    delay = max(delay, float(retry_after))
                _LOGGER.warning(
                    "Agate %s failed (attempt %d); asynchronously retrying in %.1f seconds: %s: %s",
                    method,
                    error_count,
                    delay,
                    type(error).__name__,
                    error,
                )
                await (self._async_sleeper or anyio.sleep)(delay)

    @property
    def wrapped_client(self) -> object:
        """Expose the SDK client for diagnostics without bypassing the policy."""
        return self._client

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        # The SDK creates a fresh key for each invocation if absent. Keep one
        # across outer transport retries too (durable callers supply their key).
        if not request.get("idempotency_key"):
            request = {**request, "idempotency_key": uuid4().hex}
        return self._call("submit_job", lambda: self._client.submit_job(kind, request))  # type: ignore[attr-defined]

    def prepare_uploads(
        self, gpu: str, files: list[dict[str, object]], *, kind: str = "dev"
    ) -> dict[str, object]:
        return self._call(
            "prepare_uploads",
            lambda: self._client.prepare_uploads(gpu, files, kind=kind),  # type: ignore[attr-defined]
        )

    def upload_file(self, url: str, path: str) -> None:
        self._call("upload_file", lambda: self._client.upload_file(url, path))  # type: ignore[attr-defined]

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
            check_execution()
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
                    retry_state = f"{error_count}/{_EXPONENTIAL_BACKOFF_ERROR_LIMIT}"
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
