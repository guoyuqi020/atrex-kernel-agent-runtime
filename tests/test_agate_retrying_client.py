"""Tests for the uniform Agate SDK retry boundary."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from atrex_runtime.gateway.retrying_client import RetryingAgateClient


@dataclass
class FlakyHealthClient:
    failures_before_success: int
    calls: int = 0

    def health(self) -> bool:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise ConnectionError(f"temporary failure {self.calls}")
        return True


def test_retrying_agate_client_uses_exponential_backoff_until_success() -> None:
    raw = FlakyHealthClient(failures_before_success=3)
    delays: list[float] = []
    client = RetryingAgateClient(raw, sleeper=delays.append)

    assert client.health() is True
    assert raw.calls == 4
    assert delays == [5.0, 10.0, 20.0]


def test_retrying_agate_client_enters_steady_retry_after_five_errors() -> None:
    raw = FlakyHealthClient(failures_before_success=5)
    delays: list[float] = []
    client = RetryingAgateClient(raw, sleeper=delays.append)

    assert client.health() is True

    assert raw.calls == 6
    assert delays == [5.0, 10.0, 20.0, 40.0, 60.0]


def test_retrying_agate_client_retries_every_minute_until_recovery() -> None:
    raw = FlakyHealthClient(failures_before_success=8)
    delays: list[float] = []
    client = RetryingAgateClient(raw, sleeper=delays.append)

    assert client.health() is True

    assert raw.calls == 9
    assert delays == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0, 60.0]


def test_success_starts_the_next_request_with_a_fresh_error_count() -> None:
    raw = FlakyHealthClient(failures_before_success=1)
    delays: list[float] = []
    client = RetryingAgateClient(raw, sleeper=delays.append)

    assert client.health() is True
    raw.failures_before_success = raw.calls + 1
    assert client.health() is True

    assert raw.calls == 4
    assert delays == [5.0, 5.0]


class StatusError(Exception):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"[{status}] source validation failed")


@dataclass
class StatusClient:
    status: int
    failures_before_success: int | None = None
    calls: int = 0

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        if self.failures_before_success is None or self.calls <= self.failures_before_success:
            raise StatusError(self.status)
        return {"status": "queued", "job_id": "job-recovered"}


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_permanent_client_errors_are_raised_without_retrying(status: int) -> None:
    raw = StatusClient(status=status)
    delays: list[float] = []
    client = RetryingAgateClient(raw, sleeper=delays.append)

    with pytest.raises(StatusError, match="source validation failed"):
        client.submit_job("dev", {})

    assert raw.calls == 1
    assert delays == []


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_retryable_statuses_keep_the_backoff_contract(status: int) -> None:
    raw = StatusClient(status=status, failures_before_success=6)
    delays: list[float] = []
    client = RetryingAgateClient(raw, sleeper=delays.append)

    assert client.submit_job("dev", {}) == {
        "status": "queued",
        "job_id": "job-recovered",
    }

    assert raw.calls == 7
    assert delays == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
