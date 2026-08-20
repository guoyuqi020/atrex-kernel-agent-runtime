"""Background health observation for external Runtime dependencies."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable


class PeriodicHealthMonitor:
    """Run one synchronous liveness probe without blocking the ASGI event loop."""

    def __init__(
        self,
        name: str,
        probe: Callable[[], bool],
        *,
        interval_seconds: float,
        logger: logging.Logger | None = None,
    ) -> None:
        if not name:
            raise ValueError("health monitor name cannot be empty")
        if interval_seconds <= 0:
            raise ValueError("health check interval must be positive")
        self._name = name
        self._probe = probe
        self._interval_seconds = interval_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._healthy: bool | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def healthy(self) -> bool | None:
        """Return the latest observed state, or ``None`` before the first result."""
        return self._healthy

    def start(self) -> None:
        """Start probing immediately and then at the configured interval."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(
            self._run(),
            name=f"atrex-{self._name}-health",
        )

    def cancel(self) -> None:
        """Request prompt shutdown from a synchronous owner."""
        if self._task is not None:
            self._task.cancel()

    async def stop(self) -> None:
        """Cancel and join the background task."""
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        while True:
            await self._probe_once()
            await asyncio.sleep(self._interval_seconds)

    async def _probe_once(self) -> None:
        try:
            healthy = bool(await asyncio.to_thread(self._probe))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            healthy = False
            self._logger.debug(
                "%s health probe raised %s",
                self._name,
                type(error).__name__,
                exc_info=True,
            )

        previous = self._healthy
        self._healthy = healthy
        if healthy and previous is None:
            self._logger.info("%s health check succeeded", self._name)
        elif healthy and previous is False:
            self._logger.info("%s health recovered", self._name)
        elif not healthy and previous is not False:
            self._logger.warning("%s health check failed", self._name)
        else:
            self._logger.debug(
                "%s health check %s",
                self._name,
                "succeeded" if healthy else "failed",
            )
