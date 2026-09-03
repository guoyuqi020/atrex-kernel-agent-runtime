"""Shared construction helpers for Runtime tests."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass

from atrex_runtime.config import CampaignRuntimeSettings
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    KernelAgentRevisionId,
    LineageId,
    new_campaign_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
    parse_artifact_digest,
)
from atrex_runtime.domain.models import (
    Campaign,
    Dsl,
    KernelAgentRevision,
    KernelEvaluation,
    KernelRevision,
    Lineage,
    LineageStatus,
)
from atrex_runtime.kernel_agents import KernelAgentBundleLimits
from atrex_runtime.ports import BuildAttemptEvidenceRequest
from atrex_runtime.registry.sqlite import SqliteRegistry

NOW = "2026-08-14T00:00:00+00:00"


def with_local_interpreter(campaign: CampaignRuntimeSettings) -> CampaignRuntimeSettings:
    """Retarget checked-in Worker command prefixes at the interpreter running the tests.

    The committed configuration names the repository's conventional `.venv/bin/python`,
    which need not exist wherever the suite runs.
    """
    return campaign.model_copy(
        update={
            "optimizer": campaign.optimizer.model_copy(
                update={"command_prefix": (sys.executable, *campaign.optimizer.command_prefix[1:])}
            ),
            "evolver": campaign.evolver.model_copy(
                update={"command_prefix": (sys.executable, *campaign.evolver.command_prefix[1:])}
            ),
        }
    )


def kernel_agent_limits(*, max_entrypoint_bytes: int = 8192) -> KernelAgentBundleLimits:
    """Return explicit small Bundle limits for unit tests."""
    return KernelAgentBundleLimits(
        max_bundle_files=128,
        max_bundle_bytes=65_536,
        max_entrypoint_bytes=max_entrypoint_bytes,
    )


@dataclass
class FakeAttemptEvidence:
    """Return deterministic valid digests while recording branch visibility requests."""

    requests: list[BuildAttemptEvidenceRequest]

    def __init__(self) -> None:
        self.requests = []

    def assemble(self, request: BuildAttemptEvidenceRequest) -> ArtifactDigest:
        self.requests.append(request)
        return digest(f"attempt-evidence:{request.attempt_id}")

    def validate(
        self,
        _digest: ArtifactDigest,
        _request: BuildAttemptEvidenceRequest,
    ) -> None:
        """Accept the deterministic fake snapshot."""


def digest(label: str) -> ArtifactDigest:
    """Return a valid deterministic digest for one test label."""
    hexadecimal = hashlib.sha256(label.encode()).hexdigest()
    return parse_artifact_digest(f"sha256:{hexadecimal}")


@dataclass(frozen=True, slots=True)
class SeededLineage:
    """Identifiers and baseline records created for one test lineage."""

    lineage_id: LineageId
    active_revision_id: KernelAgentRevisionId
    baseline: KernelRevision


def seed_lineage(
    registry: SqliteRegistry,
    *,
    dsl: Dsl = Dsl.TRITON,
    evaluation_contract_digest: ArtifactDigest | None = None,
    kernel_artifact_digest: ArtifactDigest | None = None,
    gateway_result_digest: ArtifactDigest | None = None,
    evidence_checkpoint: ArtifactDigest | None = None,
    challenger_count: int = 1,
    challenger_start_epoch: int = 1,
    first_epoch_same_agent: bool = False,
    trajectories_per_branch: int = 1,
    attempts_per_trajectory: int = 2,
    optimizer_model: str | None = None,
    evolver_model: str | None = None,
    ephemeral_agent_state: bool = False,
    bootstrap_source_lineage_id: LineageId | None = None,
) -> SeededLineage:
    """Create a complete ready lineage with one correct baseline Kernel."""
    campaign_id = new_campaign_id()
    registry.insert_campaign(
        Campaign(
            id=campaign_id,
            operator="vector_add",
            hardware_target="nvidia-h100",
            evaluation_contract_digest=(
                evaluation_contract_digest or digest("evaluation-contract")
            ),
            agent_problem_digest=digest("agent-problem"),
            created_at=NOW,
        )
    )
    active_revision_id = new_kernel_agent_revision_id()
    registry.register_kernel_agent_revision(
        KernelAgentRevision(
            id=active_revision_id,
            parent_id=None,
            creation_key=f"bootstrap:{dsl.value}",
            dsl=dsl,
            optimizer_digest=digest("active-optimizer"),
            created_by="bootstrap",
            created_at=NOW,
            source_provenance_digest=digest("active-source"),
        )
    )
    baseline = KernelRevision(
        id=new_kernel_revision_id(),
        parent_id=None,
        artifact_digest=kernel_artifact_digest or digest("baseline-kernel"),
        produced_by_attempt_id=None,
        evaluation=KernelEvaluation(
            correct=True,
            latency_us=100.0,
            gateway_result_digest=gateway_result_digest or digest("baseline-gateway"),
        ),
        created_at=NOW,
    )
    registry.register_kernel_revision(baseline)
    lineage_id = new_lineage_id()
    registry.insert_lineage(
        Lineage(
            id=lineage_id,
            campaign_id=campaign_id,
            dsl=dsl,
            hardware_target="nvidia-h100",
            active_kernel_agent_revision_id=active_revision_id,
            best_kernel_revision_id=baseline.id,
            evidence_checkpoint=evidence_checkpoint or digest("epoch-1-evidence"),
            challenger_count=challenger_count,
            challenger_start_epoch=challenger_start_epoch,
            first_epoch_same_agent=first_epoch_same_agent,
            trajectories_per_branch=trajectories_per_branch,
            attempts_per_trajectory=attempts_per_trajectory,
            next_epoch_number=1,
            status=LineageStatus.READY,
            optimizer_model=optimizer_model,
            evolver_model=evolver_model,
            ephemeral_agent_state=ephemeral_agent_state,
            bootstrap_source_lineage_id=bootstrap_source_lineage_id,
        )
    )
    return SeededLineage(lineage_id, active_revision_id, baseline)
