"""Durable Campaign task leasing and execution."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol

import anyio

from ..domain.ids import CampaignId, CampaignTaskId
from ..domain.models import CampaignTask, CampaignTaskStatus
from ..registry.base import Registry


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RegisteredCampaignScheduler(Protocol):
    """Scheduling operation required by a Campaign task worker."""

    async def run_registered_campaign_through(
        self,
        campaign_id: CampaignId,
        target_epoch_number: int,
        *,
        finalize: bool = False,
    ) -> object:
        """Run one registered Campaign through an absolute target."""
        ...


class CampaignTaskWorker:
    """Lease and execute durable Campaign tasks outside the ASGI process."""

    def __init__(
        self,
        registry: Registry,
        scheduler: RegisteredCampaignScheduler,
        *,
        owner: str,
        lease_seconds: float,
        heartbeat_seconds: float,
        max_error_bytes: int,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not owner:
            raise ValueError("Campaign task worker owner cannot be empty")
        if lease_seconds <= heartbeat_seconds * 2:
            raise ValueError("Campaign task lease must exceed two heartbeat periods")
        if max_error_bytes <= 0:
            raise ValueError("Campaign task error limit must be positive")
        self._registry = registry
        self._scheduler = scheduler
        self._owner = owner
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._max_error_bytes = max_error_bytes
        self._clock = clock

    async def run_once(self) -> CampaignTask | None:
        """Claim and execute at most one ready task."""
        now = self._clock()
        task = self._registry.claim_campaign_task(
            self._owner,
            now=now.isoformat(),
            lease_expires_at=(now + timedelta(seconds=self._lease_seconds)).isoformat(),
        )
        if task is None:
            return None
        if task.status is CampaignTaskStatus.CANCELLED:
            return task

        stopped = anyio.Event()
        failure: Exception | None = None
        cancellation = anyio.CancelScope()
        async with anyio.create_task_group() as workers:
            workers.start_soon(self._heartbeat, task.id, stopped, cancellation)
            with cancellation:
                try:
                    await self._scheduler.run_registered_campaign_through(
                        task.campaign_id,
                        task.target_epoch_number,
                        finalize=task.finalize,
                    )
                except Exception as error:
                    failure = error
                finally:
                    stopped.set()

        if failure is None:
            return self._registry.complete_campaign_task(task.id, self._owner)
        return self._registry.fail_campaign_task(
            task.id,
            self._owner,
            error=self._bounded_error(failure),
        )

    async def _heartbeat(
        self,
        task_id: CampaignTaskId,
        stopped: anyio.Event,
        cancellation: anyio.CancelScope,
    ) -> None:
        while True:
            with anyio.move_on_after(self._heartbeat_seconds):
                await stopped.wait()
            if stopped.is_set():
                return
            now = self._clock()
            cancel_requested = self._registry.renew_campaign_task(
                task_id,
                self._owner,
                lease_expires_at=(now + timedelta(seconds=self._lease_seconds)).isoformat(),
            )
            if cancel_requested:
                cancellation.cancel()
                return

    def _bounded_error(self, error: Exception) -> str:
        value = f"{type(error).__name__}: {error}"
        payload = value.encode("utf-8")[: self._max_error_bytes]
        return payload.decode("utf-8", errors="ignore") or type(error).__name__
