"""Recoverable scheduling across independent DSL lineages."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol

import anyio

from ..domain.errors import InvalidTransitionError
from ..domain.ids import ArtifactDigest, CampaignId, LineageId
from ..domain.models import CampaignStatus, Lineage, LineageStatus
from ..registry.base import Registry
from .epoch import EpochController
from .leases import LineageLeaseManager


class EvidenceAssembler(Protocol):
    """Build the cumulative checkpoint after one completed epoch."""

    def assemble_next(
        self,
        lineage_id: LineageId,
    ) -> ArtifactDigest:
        """Return a deterministic checkpoint for the lineage handoff state."""
        ...


@dataclass(frozen=True, slots=True)
class LineageScheduleResult:
    """Final durable state after running one lineage through a target epoch."""

    lineage: Lineage
    completed_epochs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CampaignScheduleResult:
    """Ordered final states for one parallel multi-DSL scheduling request."""

    campaign_id: CampaignId
    lineages: tuple[LineageScheduleResult, ...]


class CampaignScheduler:
    """Drive idempotent epoch targets while preserving Evidence handoffs."""

    def __init__(
        self,
        registry: Registry,
        epochs: EpochController,
        evidence: EvidenceAssembler,
        leases: LineageLeaseManager,
        evolver_commit: str | None = None,
    ) -> None:
        self._registry = registry
        self._epochs = epochs
        self._evidence = evidence
        self._leases = leases
        self._evolver_commit = evolver_commit

    async def run_lineage_through(
        self,
        lineage_id: LineageId,
        target_epoch_number: int,
    ) -> LineageScheduleResult:
        """Create or resume epochs until the durable target has been completed."""
        if target_epoch_number <= 0:
            raise ValueError("target_epoch_number must be positive")
        with self._leases.acquire(lineage_id):
            if self._evolver_commit is not None:
                lineage = self._registry.get_lineage(lineage_id)
                self._registry.ensure_campaign_evolver_commit(
                    lineage.campaign_id,
                    self._evolver_commit,
                )
            return await self._run_owned_lineage_through(lineage_id, target_epoch_number)

    async def _run_owned_lineage_through(
        self,
        lineage_id: LineageId,
        target_epoch_number: int,
    ) -> LineageScheduleResult:
        """Advance a lineage while the caller holds its process-lifetime lease."""
        completed: list[int] = []
        while True:
            lineage = self._registry.get_lineage(lineage_id)
            if lineage.status is LineageStatus.AWAITING_EVIDENCE:
                previous = lineage.evidence_checkpoint
                next_checkpoint = await anyio.to_thread.run_sync(
                    self._evidence.assemble_next,
                    lineage_id,
                )
                self._registry.advance_lineage_evidence(
                    lineage_id,
                    previous,
                    next_checkpoint,
                )
                continue
            if lineage.status is LineageStatus.FAILED:
                raise InvalidTransitionError(f"Lineage {lineage_id} has failed")
            if lineage.status is LineageStatus.CANCELLED:
                raise InvalidTransitionError(f"Lineage {lineage_id} is cancelled")
            if lineage.status is LineageStatus.COMPLETED:
                if lineage.next_epoch_number <= target_epoch_number:
                    raise InvalidTransitionError(
                        f"Lineage {lineage_id} completed before target epoch"
                    )
                return LineageScheduleResult(lineage, tuple(completed))
            if lineage.next_epoch_number > target_epoch_number:
                if lineage.status is not LineageStatus.READY:
                    raise InvalidTransitionError(
                        f"Lineage {lineage_id} stopped in {lineage.status}"
                    )
                return LineageScheduleResult(lineage, tuple(completed))
            result = await self._epochs.run_epoch(lineage_id, lineage.next_epoch_number)
            completed.append(result.epoch.number)

    async def run_campaign_through(
        self,
        lineage_ids: tuple[LineageId, ...],
        target_epoch_number: int,
    ) -> CampaignScheduleResult:
        """Run distinct lineages concurrently through the same epoch number."""
        if target_epoch_number <= 0:
            raise ValueError("target_epoch_number must be positive")
        if not lineage_ids:
            raise ValueError("a Campaign schedule requires at least one lineage")
        if len(set(lineage_ids)) != len(lineage_ids):
            raise ValueError("a Campaign schedule cannot repeat a lineage")
        lineages = [self._registry.get_lineage(lineage_id) for lineage_id in lineage_ids]
        campaign_id = lineages[0].campaign_id
        if any(lineage.campaign_id != campaign_id for lineage in lineages[1:]):
            raise ValueError("all scheduled lineages must belong to one Campaign")

        results: dict[LineageId, LineageScheduleResult] = {}

        async def run_one(lineage_id: LineageId) -> None:
            results[lineage_id] = await self.run_lineage_through(
                lineage_id,
                target_epoch_number,
            )

        async with anyio.create_task_group() as tasks:
            for lineage_id in lineage_ids:
                tasks.start_soon(run_one, lineage_id)
        return CampaignScheduleResult(
            campaign_id,
            tuple(results[lineage_id] for lineage_id in lineage_ids),
        )

    async def run_registered_campaign_through(
        self,
        campaign_id: CampaignId,
        target_epoch_number: int,
        *,
        finalize: bool = False,
    ) -> CampaignScheduleResult:
        """Discover every registered lineage, run it, and optionally finalize the Campaign."""
        campaign = self._registry.get_campaign(campaign_id)
        if campaign.status is not CampaignStatus.ACTIVE:
            raise InvalidTransitionError(f"Campaign {campaign_id} is {campaign.status}")
        lineages = self._registry.list_campaign_lineages(campaign_id)
        if not lineages:
            raise InvalidTransitionError(f"Campaign {campaign_id} has no lineages")
        result = await self.run_campaign_through(
            tuple(lineage.id for lineage in lineages),
            target_epoch_number,
        )
        if not finalize:
            return result
        with ExitStack() as held:
            for lineage in lineages:
                held.enter_context(self._leases.acquire(lineage.id))
            self._registry.complete_campaign(campaign_id, target_epoch_number)
        return CampaignScheduleResult(
            campaign_id,
            tuple(
                LineageScheduleResult(
                    self._registry.get_lineage(item.lineage.id),
                    item.completed_epochs,
                )
                for item in result.lineages
            ),
        )
