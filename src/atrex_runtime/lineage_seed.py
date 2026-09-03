"""Create an independent Lineage from sealed Agent and Kernel content."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from .controller.evidence import LocalEvidenceAssembler
from .domain.ids import (
    ArtifactDigest,
    CampaignId,
    KernelAgentRevisionId,
    KernelRevisionId,
    LineageId,
    parse_artifact_digest,
    parse_kernel_agent_revision_id,
    parse_kernel_revision_id,
    parse_lineage_id,
)
from .domain.models import (
    CampaignStatus,
    Dsl,
    KernelAgentRevision,
    KernelEvaluation,
    KernelRevision,
    Lineage,
    LineageStatus,
)
from .kernel_agents import KernelAgentRevisionBuilder
from .ports import AttemptCandidateResult
from .registry.base import Registry

LINEAGE_SEED_SPEC_VERSION: Literal[1] = 1
LINEAGE_SEED_PROVENANCE_VERSION: Literal[1] = 1


class ArtifactLineageSeedV1(BaseModel):
    """Exact CAS objects selected without relying on existing revision identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: Literal["artifacts"] = "artifacts"
    agent_artifact_digest: ArtifactDigest
    kernel_artifact_digest: ArtifactDigest

    @field_validator("agent_artifact_digest", "kernel_artifact_digest", mode="before")
    @classmethod
    def _validate_digest(cls, value: object) -> ArtifactDigest:
        if not isinstance(value, str):
            raise ValueError("Lineage seed Artifact digest must be a string")
        return parse_artifact_digest(value)


class RevisionLineageSeedV1(BaseModel):
    """Existing revisions whose content Artifacts seed a new independent Lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: Literal["revisions"] = "revisions"
    agent_revision_id: KernelAgentRevisionId
    kernel_revision_id: KernelRevisionId

    @field_validator("agent_revision_id", mode="before")
    @classmethod
    def _validate_agent_revision_id(cls, value: object) -> KernelAgentRevisionId:
        if not isinstance(value, str):
            raise ValueError("source Agent revision ID must be a string")
        return parse_kernel_agent_revision_id(value)

    @field_validator("kernel_revision_id", mode="before")
    @classmethod
    def _validate_kernel_revision_id(cls, value: object) -> KernelRevisionId:
        if not isinstance(value, str):
            raise ValueError("source Kernel revision ID must be a string")
        return parse_kernel_revision_id(value)


class LineageBaselineSeedV1(BaseModel):
    """Another Lineage's frozen v0 baseline, cloned whole rather than re-measured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: Literal["lineage_baseline"] = "lineage_baseline"
    lineage_id: LineageId

    @field_validator("lineage_id", mode="before")
    @classmethod
    def _validate_lineage_id(cls, value: object) -> LineageId:
        if not isinstance(value, str):
            raise ValueError("source Lineage ID must be a string")
        return parse_lineage_id(value)


LineageSeedSourceV1 = Annotated[
    ArtifactLineageSeedV1 | RevisionLineageSeedV1 | LineageBaselineSeedV1,
    Field(discriminator="source_type"),
]


class LineageSeedModelsV1(BaseModel):
    """Concrete model identities owned by the new Lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    optimizer: str | None = Field(default=None, min_length=1, max_length=200)
    evolver: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("optimizer", "evolver")
    @classmethod
    def _validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("Lineage seed model must be non-empty and cannot contain NUL")
        return normalized


class LineageSeedSpecV1(BaseModel):
    """Strict request for one Artifact- or Revision-seeded Lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = LINEAGE_SEED_SPEC_VERSION
    creation_key: str = Field(min_length=1, max_length=200)
    dsl: Dsl
    seed: LineageSeedSourceV1
    initial_evidence: Path | None = None
    models: LineageSeedModelsV1 = LineageSeedModelsV1()
    challenger_count: int = Field(default=1, ge=0)
    challenger_start_epoch: int = Field(default=1, gt=0)
    first_epoch_same_agent: bool = False
    trajectories_per_branch: int = Field(default=1, gt=0)
    attempts_per_trajectory: int = Field(gt=0)
    ephemeral_agent_state: bool = False

    @field_validator("creation_key")
    @classmethod
    def _validate_creation_key(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("Lineage seed creation_key is invalid")
        return normalized

    @model_validator(mode="after")
    def _validate_seed_combination(self) -> Self:
        if self.first_epoch_same_agent and self.challenger_count != 1:
            raise ValueError("first_epoch_same_agent requires exactly one Challenger")
        if isinstance(self.seed, LineageBaselineSeedV1) and self.initial_evidence is not None:
            raise ValueError("a cloned Lineage baseline already carries its Bootstrap Evidence")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Parse strict JSON and resolve the optional Evidence path from the spec directory."""
        source = Path(path).resolve()
        try:
            value = json.loads(source.read_bytes())
        except json.JSONDecodeError as error:
            raise ValueError("Lineage seed spec is not valid JSON") from error
        spec = cls.model_validate(value)
        evidence = spec.initial_evidence
        if evidence is not None and not evidence.is_absolute():
            spec = spec.model_copy(
                update={"initial_evidence": (source.parent / evidence).resolve()}
            )
        return spec


def parse_lineage_seed_spec_json(payload: bytes) -> LineageSeedSpecV1:
    """Parse an HTTP Lineage seed request without resolving server-host paths."""
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("Lineage seed request is not valid JSON") from error
    return LineageSeedSpecV1.model_validate(value)


class LineageSeedKernelEvaluator(Protocol):
    """Trusted direct evaluation used before a seed Kernel becomes v0."""

    async def evaluate(
        self,
        *,
        campaign_id: CampaignId,
        lineage_id: LineageId,
        dsl: Dsl,
        kernel_artifact_digest: ArtifactDigest,
    ) -> AttemptCandidateResult:
        """Evaluate the exact Kernel under the target Campaign contract."""


@dataclass(frozen=True, slots=True)
class LineageSeedResult:
    """New or idempotently recovered Lineage root identities."""

    campaign_id: CampaignId
    lineage_id: LineageId
    dsl: Dsl
    kernel_agent_revision_id: KernelAgentRevisionId
    kernel_revision_id: KernelRevisionId
    agent_artifact_digest: ArtifactDigest
    kernel_artifact_digest: ArtifactDigest
    gateway_result_digest: ArtifactDigest
    latency_us: float
    evidence_checkpoint: ArtifactDigest
    source_provenance_digest: ArtifactDigest
    source_agent_revision_id: KernelAgentRevisionId | None
    source_kernel_revision_id: KernelRevisionId | None
    optimizer_model: str | None
    evolver_model: str | None
    created_at: str


class LineageSeeder:
    """Validate sealed roots, re-evaluate the Kernel, and publish a fresh Lineage tree."""

    def __init__(
        self,
        registry: Registry,
        artifacts: LocalArtifactStore,
        agent_builder: KernelAgentRevisionBuilder,
        evaluator: LineageSeedKernelEvaluator,
        *,
        evolver_commit: str | None = None,
        clock: Callable[[], str] = lambda: datetime.now(UTC).isoformat(),
    ) -> None:
        self._registry = registry
        self._artifacts = artifacts
        self._agent_builder = agent_builder
        self._evaluator = evaluator
        self._evolver_commit = evolver_commit
        self._clock = clock

    async def seed_lineage(
        self,
        campaign_id: CampaignId,
        spec: LineageSeedSpecV1,
    ) -> LineageSeedResult:
        """Create or recover one independent Lineage rooted at the selected content."""
        campaign = self._registry.get_campaign(campaign_id)
        if self._evolver_commit is not None:
            campaign = self._registry.ensure_campaign_evolver_commit(
                campaign_id,
                self._evolver_commit,
            )
        if campaign.status is not CampaignStatus.ACTIVE:
            raise ValueError("new Lineages can be added only to an active Campaign")
        lineage_id = parse_lineage_id(
            self._derived_id("lineage", f"{campaign_id}:{spec.creation_key}")
        )
        agent_id = parse_kernel_agent_revision_id(
            self._derived_id("agentrev", f"{lineage_id}:seed-agent")
        )
        kernel_id = parse_kernel_revision_id(
            self._derived_id("kernelrev", f"{lineage_id}:seed-kernel")
        )
        try:
            existing = self._registry.get_lineage(lineage_id)
        except KeyError:
            existing = None
        roots = self._resolve_roots(spec.seed, spec.dsl)
        evidence = self._initial_evidence(
            lineage_id,
            spec.initial_evidence,
            roots.bootstrap_evidence_source,
        )
        provenance_digest = self._provenance(
            campaign_id,
            lineage_id,
            spec,
            roots,
            evidence,
        )
        if existing is not None:
            return self._validate_existing(
                existing,
                campaign_id,
                spec,
                roots,
                provenance_digest,
                evidence,
            )

        self._validate_agent_artifact(roots.agent_artifact_digest, spec.dsl)
        try:
            kernel = self._registry.get_kernel_revision(kernel_id)
        except KeyError:
            kernel = None
        if kernel is None:
            if roots.baseline_evaluation is not None:
                created_at = self._clock()
                kernel = KernelRevision(
                    id=kernel_id,
                    parent_id=None,
                    artifact_digest=roots.kernel_artifact_digest,
                    produced_by_attempt_id=None,
                    evaluation=roots.baseline_evaluation,
                    created_at=created_at,
                )
                self._registry.register_kernel_revision(kernel)
            else:
                outcome = await self._evaluator.evaluate(
                    campaign_id=campaign_id,
                    lineage_id=lineage_id,
                    dsl=spec.dsl,
                    kernel_artifact_digest=roots.kernel_artifact_digest,
                )
                if outcome.artifact_digest != roots.kernel_artifact_digest:
                    raise ValueError("Lineage seed evaluation returned a different Kernel Artifact")
                if not outcome.correct or outcome.latency_us is None:
                    raise ValueError("Lineage seed Kernel failed authoritative evaluation")
                if (
                    self._artifacts.verify(outcome.gateway_result_digest).kind
                    is not ArtifactKind.GATEWAY_RESULT
                ):
                    raise ValueError("Lineage seed evaluation result has the wrong Artifact kind")
                created_at = self._clock()
                kernel = KernelRevision(
                    id=kernel_id,
                    parent_id=None,
                    artifact_digest=roots.kernel_artifact_digest,
                    produced_by_attempt_id=None,
                    evaluation=KernelEvaluation(
                        correct=True,
                        latency_us=outcome.latency_us,
                        gateway_result_digest=outcome.gateway_result_digest,
                    ),
                    created_at=created_at,
                )
                self._registry.register_kernel_revision(kernel)
        else:
            if kernel.artifact_digest != roots.kernel_artifact_digest:
                raise ValueError("Lineage seed Kernel identity resolved to different content")
            if not kernel.evaluation.correct or kernel.evaluation.latency_us is None:
                raise ValueError("Recovered Lineage seed Kernel is not correct")
            created_at = kernel.created_at

        agent = self._registry.register_kernel_agent_revision(
            KernelAgentRevision(
                id=agent_id,
                parent_id=None,
                creation_key=f"lineage-seed:{campaign_id}:{spec.creation_key}",
                dsl=spec.dsl,
                optimizer_digest=roots.agent_artifact_digest,
                created_by="lineage_seed",
                created_at=created_at,
                source_provenance_digest=provenance_digest,
            )
        )
        lineage = Lineage(
            id=lineage_id,
            campaign_id=campaign_id,
            dsl=spec.dsl,
            hardware_target=campaign.hardware_target,
            active_kernel_agent_revision_id=agent.id,
            best_kernel_revision_id=kernel.id,
            evidence_checkpoint=evidence,
            challenger_count=spec.challenger_count,
            challenger_start_epoch=spec.challenger_start_epoch,
            first_epoch_same_agent=spec.first_epoch_same_agent,
            trajectories_per_branch=spec.trajectories_per_branch,
            attempts_per_trajectory=spec.attempts_per_trajectory,
            next_epoch_number=1,
            status=LineageStatus.READY,
            optimizer_model=spec.models.optimizer,
            evolver_model=spec.models.evolver,
            ephemeral_agent_state=spec.ephemeral_agent_state,
            bootstrap_source_lineage_id=roots.source_lineage_id,
        )
        self._registry.insert_lineage(lineage)
        self._registry.record_runtime_event(
            "lineage.seeded",
            lineage.id,
            {
                "campaign_id": campaign_id,
                "dsl": spec.dsl.value,
                "agent_artifact_digest": roots.agent_artifact_digest,
                "kernel_artifact_digest": roots.kernel_artifact_digest,
                "source_provenance_digest": provenance_digest,
            },
        )
        return self._result(lineage, agent, kernel, provenance_digest, roots, evidence)

    def _resolve_roots(
        self,
        source: LineageSeedSourceV1,
        dsl: Dsl,
    ) -> _ResolvedSeedRoots:
        if isinstance(source, LineageBaselineSeedV1):
            return self._resolve_lineage_baseline(source.lineage_id, dsl)
        if isinstance(source, ArtifactLineageSeedV1):
            agent_digest = source.agent_artifact_digest
            kernel_digest = source.kernel_artifact_digest
            agent_revision_id = None
            kernel_revision_id = None
        else:
            agent = self._registry.get_kernel_agent_revision(source.agent_revision_id)
            kernel = self._registry.get_kernel_revision(source.kernel_revision_id)
            if agent.dsl is not dsl:
                raise ValueError("source Agent revision belongs to a different DSL")
            if self._registry.find_kernel_lineage(kernel.id).dsl is not dsl:
                raise ValueError("source Kernel revision belongs to a different DSL")
            agent_digest = agent.optimizer_digest
            kernel_digest = kernel.artifact_digest
            agent_revision_id = agent.id
            kernel_revision_id = kernel.id
        if self._artifacts.verify(agent_digest).kind is not ArtifactKind.KERNEL_AGENT:
            raise ValueError("Lineage seed Agent Artifact has the wrong kind")
        if self._artifacts.verify(kernel_digest).kind is not ArtifactKind.KERNEL:
            raise ValueError("Lineage seed Kernel Artifact has the wrong kind")
        return _ResolvedSeedRoots(
            agent_digest,
            kernel_digest,
            agent_revision_id,
            kernel_revision_id,
        )

    def _resolve_lineage_baseline(
        self,
        source_lineage_id: LineageId,
        dsl: Dsl,
    ) -> _ResolvedSeedRoots:
        source = self._registry.get_lineage(source_lineage_id)
        if source.dsl is not dsl:
            raise ValueError("source Lineage belongs to a different DSL")
        agent_entries = self._registry.list_lineage_agent_revisions(source_lineage_id)
        kernel_entries = self._registry.list_lineage_kernels(source_lineage_id)
        if not agent_entries or agent_entries[0].revision_number != 0:
            raise ValueError("source Lineage has no agent-v0 baseline")
        if not kernel_entries or kernel_entries[0].revision_number != 0:
            raise ValueError("source Lineage has no v0 Kernel baseline")
        agent = agent_entries[0].revision
        kernel = kernel_entries[0].revision
        if not kernel.evaluation.correct or kernel.evaluation.latency_us is None:
            raise ValueError("source Lineage baseline Kernel has no correct measurement")
        if self._artifacts.verify(agent.optimizer_digest).kind is not ArtifactKind.KERNEL_AGENT:
            raise ValueError("Lineage seed Agent Artifact has the wrong kind")
        if self._artifacts.verify(kernel.artifact_digest).kind is not ArtifactKind.KERNEL:
            raise ValueError("Lineage seed Kernel Artifact has the wrong kind")
        return _ResolvedSeedRoots(
            agent.optimizer_digest,
            kernel.artifact_digest,
            agent.id,
            kernel.id,
            source_lineage_id=source_lineage_id,
            baseline_evaluation=kernel.evaluation,
            bootstrap_evidence_source=source.evidence_checkpoint,
        )

    def _validate_agent_artifact(self, digest: ArtifactDigest, dsl: Dsl) -> None:
        self._artifacts.verify(digest)
        with tempfile.TemporaryDirectory(prefix="atrex-lineage-seed-agent-") as temporary:
            root = Path(temporary) / "agent"
            self._artifacts.materialize(digest, root)
            candidate = self._agent_builder.build_candidate(root, dsl)
        if candidate.optimizer_digest != digest:
            raise ValueError("Lineage seed Agent Artifact changed during validation")

    def _initial_evidence(
        self,
        lineage_id: LineageId,
        source: Path | None,
        bootstrap_evidence_source: ArtifactDigest | None = None,
    ) -> ArtifactDigest:
        assembler = LocalEvidenceAssembler(self._registry, self._artifacts)
        if bootstrap_evidence_source is not None:
            return assembler.clone_bootstrap(lineage_id, bootstrap_evidence_source)
        if source is not None:
            return assembler.create_initial(
                lineage_id,
                source,
                source_label="lineage-seed-input",
            )
        with tempfile.TemporaryDirectory(prefix="atrex-lineage-seed-evidence-") as temporary:
            return assembler.create_initial(
                lineage_id,
                temporary,
                source_label="empty-lineage-seed",
            )

    def _provenance(
        self,
        campaign_id: CampaignId,
        lineage_id: LineageId,
        spec: LineageSeedSpecV1,
        roots: _ResolvedSeedRoots,
        evidence_checkpoint: ArtifactDigest,
    ) -> ArtifactDigest:
        value: JsonValue = {
            "schema_version": LINEAGE_SEED_PROVENANCE_VERSION,
            "source_type": "lineage_seed",
            "campaign_id": campaign_id,
            "lineage_id": lineage_id,
            "creation_key": spec.creation_key,
            "dsl": spec.dsl.value,
            "agent_artifact_digest": roots.agent_artifact_digest,
            "kernel_artifact_digest": roots.kernel_artifact_digest,
            "source_agent_revision_id": roots.source_agent_revision_id,
            "source_kernel_revision_id": roots.source_kernel_revision_id,
            "source_lineage_id": roots.source_lineage_id,
            "ephemeral_agent_state": spec.ephemeral_agent_state,
            "initial_evidence_digest": evidence_checkpoint,
        }
        return self._artifacts.put_json(value, ArtifactKind.OPTIMIZER_SOURCE)

    def _validate_existing(
        self,
        lineage: Lineage,
        campaign_id: CampaignId,
        spec: LineageSeedSpecV1,
        roots: _ResolvedSeedRoots,
        provenance_digest: ArtifactDigest,
        evidence_checkpoint: ArtifactDigest,
    ) -> LineageSeedResult:
        if lineage.campaign_id != campaign_id:
            raise ValueError("Lineage seed creation_key resolved to a different Campaign")
        agent_entries = self._registry.list_lineage_agent_revisions(lineage.id)
        kernel_entries = self._registry.list_lineage_kernels(lineage.id)
        if not agent_entries or agent_entries[0].revision_number != 0:
            raise ValueError("existing Lineage has no agent-v0 root")
        if not kernel_entries or kernel_entries[0].revision_number != 0:
            raise ValueError("existing Lineage has no v0 Kernel root")
        agent = agent_entries[0].revision
        kernel = kernel_entries[0].revision
        actual = (
            lineage.dsl,
            lineage.challenger_count,
            lineage.challenger_start_epoch,
            lineage.first_epoch_same_agent,
            lineage.trajectories_per_branch,
            lineage.attempts_per_trajectory,
            lineage.optimizer_model,
            lineage.evolver_model,
            lineage.ephemeral_agent_state,
            lineage.bootstrap_source_lineage_id,
            agent.optimizer_digest,
            agent.source_provenance_digest,
            kernel.artifact_digest,
        )
        expected = (
            spec.dsl,
            spec.challenger_count,
            spec.challenger_start_epoch,
            spec.first_epoch_same_agent,
            spec.trajectories_per_branch,
            spec.attempts_per_trajectory,
            spec.models.optimizer,
            spec.models.evolver,
            spec.ephemeral_agent_state,
            roots.source_lineage_id,
            roots.agent_artifact_digest,
            provenance_digest,
            roots.kernel_artifact_digest,
        )
        if actual != expected:
            raise ValueError("Lineage seed creation_key resolved to different content")
        return self._result(
            lineage,
            agent,
            kernel,
            provenance_digest,
            roots,
            evidence_checkpoint,
        )

    @staticmethod
    def _result(
        lineage: Lineage,
        agent: KernelAgentRevision,
        kernel: KernelRevision,
        provenance_digest: ArtifactDigest,
        roots: _ResolvedSeedRoots,
        evidence_checkpoint: ArtifactDigest,
    ) -> LineageSeedResult:
        latency = kernel.evaluation.latency_us
        if latency is None:
            raise AssertionError("Lineage seed result requires correct latency")
        return LineageSeedResult(
            campaign_id=lineage.campaign_id,
            lineage_id=lineage.id,
            dsl=lineage.dsl,
            kernel_agent_revision_id=agent.id,
            kernel_revision_id=kernel.id,
            agent_artifact_digest=agent.optimizer_digest,
            kernel_artifact_digest=kernel.artifact_digest,
            gateway_result_digest=kernel.evaluation.gateway_result_digest,
            latency_us=latency,
            evidence_checkpoint=evidence_checkpoint,
            source_provenance_digest=provenance_digest,
            source_agent_revision_id=roots.source_agent_revision_id,
            source_kernel_revision_id=roots.source_kernel_revision_id,
            optimizer_model=lineage.optimizer_model,
            evolver_model=lineage.evolver_model,
            created_at=kernel.created_at,
        )

    @staticmethod
    def _derived_id(prefix: str, key: str) -> str:
        suffix = hashlib.sha256(f"atrex-lineage-seed:{prefix}:{key}".encode()).hexdigest()[:32]
        return f"{prefix}_{suffix}"


@dataclass(frozen=True, slots=True)
class _ResolvedSeedRoots:
    agent_artifact_digest: ArtifactDigest
    kernel_artifact_digest: ArtifactDigest
    source_agent_revision_id: KernelAgentRevisionId | None
    source_kernel_revision_id: KernelRevisionId | None
    source_lineage_id: LineageId | None = None
    baseline_evaluation: KernelEvaluation | None = None
    bootstrap_evidence_source: ArtifactDigest | None = None
