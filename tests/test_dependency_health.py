"""Periodic external-dependency health monitor tests."""

from __future__ import annotations

import asyncio
import logging

import pytest

from atrex_runtime.dependency_health import PeriodicHealthMonitor


@pytest.mark.anyio
async def test_monitor_checks_immediately_repeats_and_stops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    results = iter((False, True, True))
    calls = 0

    def probe() -> bool:
        nonlocal calls
        calls += 1
        return next(results, True)

    monitor = PeriodicHealthMonitor("Agate", probe, interval_seconds=0.01)
    with caplog.at_level(logging.INFO):
        monitor.start()
        for _ in range(100):
            if calls >= 2:
                break
            await asyncio.sleep(0.01)
        await monitor.stop()

    assert calls >= 2
    assert monitor.healthy is True
    assert "Agate health check failed" in caplog.text
    assert "Agate health recovered" in caplog.text


@pytest.mark.anyio
async def test_monitor_treats_probe_exception_as_unhealthy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def probe() -> bool:
        raise OSError("offline")

    monitor = PeriodicHealthMonitor("Agate", probe, interval_seconds=60)
    with caplog.at_level(logging.WARNING):
        monitor.start()
        for _ in range(100):
            if monitor.healthy is not None:
                break
            await asyncio.sleep(0.01)
        await monitor.stop()

    assert monitor.healthy is False
    assert "Agate health check failed" in caplog.text


def test_monitor_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="interval must be positive"):
        PeriodicHealthMonitor("Agate", lambda: True, interval_seconds=0)
