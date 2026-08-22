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


def test_retrying_agate_client_raises_after_five_consecutive_errors() -> None:
    raw = FlakyHealthClient(failures_before_success=5)
    delays: list[float] = []
    client = RetryingAgateClient(raw, sleeper=delays.append)

    with pytest.raises(ConnectionError, match="temporary failure 5"):
        client.health()

    assert raw.calls == 5
    assert delays == [5.0, 10.0, 20.0, 40.0]


def test_success_starts_the_next_request_with_a_fresh_error_count() -> None:
    raw = FlakyHealthClient(failures_before_success=1)
    delays: list[float] = []
    client = RetryingAgateClient(raw, sleeper=delays.append)

    assert client.health() is True
    raw.failures_before_success = raw.calls + 1
    assert client.health() is True

    assert raw.calls == 4
    assert delays == [5.0, 5.0]
