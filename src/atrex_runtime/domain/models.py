"""Immutable domain records for Runtime state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    CampaignTaskId,
    EpochId,
    KernelAgentRevisionId,
    KernelRevisionId,
    LineageId,
    WorkerSessionId,
)


class Dsl(StrEnum):
    """Supported Kernel implementation languages."""

    CUDA = "cuda"
    TRITON = "triton"
    CUTEDSL = "cutedsl"


class BranchRole(StrEnum):
    """The role of one Agent branch within an epoch."""

    ACTIVE = "active"
    CHALLENGER = "challenger"


class ChallengerProposalType(StrEnum):
    """How an Evolver selected or created one Epoch Challenger."""

    EVOLVED = "evolved"
    REUSE = "reuse"
    EVOLVE_FROM_HISTORY = "evolve_from_history"


class CampaignStatus(StrEnum):
    """Lifecycle states for one multi-DSL Campaign."""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class LineageStatus(StrEnum):
    """Lifecycle states for one DSL lineage."""

    READY = "ready"
    RUNNING = "running"
    AWAITING_EVIDENCE = "awaiting_evidence"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EpochStatus(StrEnum):
    """Lifecycle states for one Active-versus-Challenger competition."""

    BUILDING_CHALLENGER = "building_challenger"
    READY = "ready"
    RUNNING = "running"
    SELECTING = "selecting"
    COMPLETED = "completed"
    FAILED = "failed"


class AttemptStatus(StrEnum):
    """Lifecycle states for one fresh Optimizer session."""

    RUNNING = "running"
    COMPLETED = "completed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


class AttemptReportStatus(StrEnum):
    """Optimizer-declared terminal state reconciled with trusted evaluation."""

    CANDIDATE_READY = "candidate_ready"
    PIVOT = "pivot"
    BLOCKED = "blocked"


class CampaignTaskStatus(StrEnum):
    """Lifecycle states for one durable Campaign scheduling request."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class KernelMeasurementPurpose(StrEnum):
    """Trusted reason for an authoritative Kernel performance measurement."""

    FRAMEWORK_BASELINE = "framework_baseline"
    ATTEMPT_EVALUATION = "attempt_evaluation"
    KERNEL_RETENTION = "kernel_retention"
    AGENT_PROMOTION = "agent_promotion"


class WorkerSessionRole(StrEnum):
    """Runtime role executed by one model-backed Worker process."""

    OPTIMIZER = "optimizer"
    FRAMEWORK_BASELINE = "framework_baseline"
    PROBLEM_GENERALIZATION = "problem_generalization"
    EVOLVER = "evolver"


class WorkerSessionStatus(StrEnum):
    """Lifecycle state of one independently launched Worker session."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-reported token buckets or Qoder-native credit consumption."""

    uncached_input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    credits: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.uncached_input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("token usage buckets must be integers")
        if any(value < 0 for value in values):
            raise ValueError("token usage buckets cannot be negative")
        if self.credits is not None:
            if (
                isinstance(self.credits, bool)
                or not isinstance(self.credits, (int, float))
                or not math.isfinite(float(self.credits))
                or self.credits < 0
            ):
                raise ValueError("credit usage must be a finite non-negative number")
            if any(values):
                raise ValueError("credit usage cannot also declare token consumption")

    @property
    def total_tokens(self) -> int:
        """Return total billed consumption without double-counting reasoning tokens."""
        return (
            self.uncached_input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def usage_unit(self) -> str:
        """Return the provider-native accounting unit."""
        return "credits" if self.credits is not None else "provider_tokens"

    @property
    def consumed(self) -> float | int:
        """Return consumption in the provider-native accounting unit."""
        return self.credits if self.credits is not None else self.total_tokens


@dataclass(frozen=True, slots=True)
class WorkerSession:
    """Durable lifecycle and raw-trace index for one Worker process."""

    id: WorkerSessionId
    role: WorkerSessionRole
    subject_id: str
    external_run_id: str
    workspace_path: str
    status: WorkerSessionStatus
    started_at: str
    campaign_id: CampaignId | None = None
    lineage_id: LineageId | None = None
    epoch_id: EpochId | None = None
    attempt_id: AttemptId | None = None
    recovery_generation: int | None = None
    backend: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    trace_digest: ArtifactDigest | None = None
    token_budget: int | None = None
    token_usage: TokenUsage | None = None
    process_returncode: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("subject ID", self.subject_id),
            ("external run ID", self.external_run_id),
            ("workspace path", self.workspace_path),
            ("start time", self.started_at),
        ):
            if not value or "\x00" in value:
                raise ValueError(f"Worker session {label} is invalid")
        if self.recovery_generation is not None and self.recovery_generation < 0:
            raise ValueError("Worker session recovery generation cannot be negative")
        if self.backend is not None and self.backend not in {"claude", "codex", "qodercli", "pi"}:
            raise ValueError(f"unsupported Worker session backend: {self.backend!r}")
        if self.model is not None and (not self.model.strip() or "\x00" in self.model):
            raise ValueError("Worker session model is invalid")
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError("Worker session token budget must be positive")
        terminal = self.status is not WorkerSessionStatus.RUNNING
        if terminal != (self.completed_at is not None):
            raise ValueError("only terminal Worker sessions require a completion time")
        if terminal != (self.finish_reason is not None):
            raise ValueError("only terminal Worker sessions require a finish reason")
        if self.finish_reason is not None and not self.finish_reason:
            raise ValueError("Worker session finish reason cannot be empty")
        if self.status is WorkerSessionStatus.RUNNING and any(
            value is not None
            for value in (
                self.trace_digest,
                self.token_usage,
                self.process_returncode,
                self.error_type,
                self.error_message,
            )
        ):
            raise ValueError("running Worker sessions cannot contain terminal results")


@dataclass(frozen=True, slots=True)
class Campaign:
    """A complete optimization job for one operator and hardware target."""

    id: CampaignId
    operator: str
    hardware_target: str
    evaluation_contract_digest: ArtifactDigest
    agent_problem_digest: ArtifactDigest
    created_at: str
    status: CampaignStatus = CampaignStatus.ACTIVE
    problem_generalization_model: str | None = None
    evolver_commit: str | None = None

    def __post_init__(self) -> None:
        if self.problem_generalization_model is not None and (
            not self.problem_generalization_model.strip()
            or "\x00" in self.problem_generalization_model
        ):
            raise ValueError("Campaign Problem Generalization model is invalid")
        if self.evolver_commit is not None and (
            len(self.evolver_commit) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in self.evolver_commit)
        ):
            raise ValueError("Campaign Evolver commit must be a full lowercase SHA")


@dataclass(frozen=True, slots=True)
class KernelAgentRevision:
    """Immutable full-repository Optimizer revision for one DSL lineage."""

    id: KernelAgentRevisionId
    parent_id: KernelAgentRevisionId | None
    creation_key: str
    dsl: Dsl
    optimizer_digest: ArtifactDigest
    created_by: str
    created_at: str
    source_provenance_digest: ArtifactDigest | None = None
    evolution_trace_digest: ArtifactDigest | None = None
    runtime_state_digest: ArtifactDigest | None = None

    def __post_init__(self) -> None:
        if self.created_by not in {"bootstrap", "lineage_seed", "evolver"}:
            raise ValueError(f"unsupported revision creator: {self.created_by!r}")
        if self.created_by in {"bootstrap", "lineage_seed"}:
            if self.source_provenance_digest is None or self.evolution_trace_digest is not None:
                raise ValueError(
                    "bootstrap and Lineage-seed revisions require only source provenance"
                )
        elif self.source_provenance_digest is not None or self.evolution_trace_digest is None:
            raise ValueError("Evolver revisions require only an Evolution trace")


@dataclass(frozen=True, slots=True)
class KernelEvaluation:
    """Authoritative Gateway evaluation attached to a Kernel revision."""

    correct: bool
    latency_us: float | None
    gateway_result_digest: ArtifactDigest

    def __post_init__(self) -> None:
        if self.correct and (self.latency_us is None or self.latency_us <= 0):
            raise ValueError("a correct Kernel requires a positive latency")
        if not self.correct and self.latency_us is not None:
            raise ValueError("an incorrect Kernel cannot carry a comparable latency")


@dataclass(frozen=True, slots=True)
class KernelRevision:
    """Immutable Kernel source and its authoritative evaluation."""

    id: KernelRevisionId
    parent_id: KernelRevisionId | None
    artifact_digest: ArtifactDigest
    produced_by_attempt_id: AttemptId | None
    evaluation: KernelEvaluation
    created_at: str


@dataclass(frozen=True, slots=True)
class KernelMeasurement:
    """One durable ordinary Evaluate sample used outside the Agent sandbox."""

    id: str
    kernel_revision_id: KernelRevisionId
    purpose: KernelMeasurementPurpose
    repeat: int
    correct: bool
    latency_us: float | None
    gateway_result_digest: ArtifactDigest | None
    agate_job_id: str | None
    created_at: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Kernel measurement ID cannot be empty")
        if self.repeat < 0:
            raise ValueError("Kernel measurement repeat cannot be negative")
        if self.correct and (self.latency_us is None or self.latency_us <= 0):
            raise ValueError("a correct Kernel measurement requires a positive latency")
        if not self.correct and self.latency_us is not None:
            raise ValueError("an incorrect Kernel measurement cannot carry a latency")
        if self.agate_job_id is not None and not self.agate_job_id:
            raise ValueError("Agate job ID cannot be empty")


@dataclass(frozen=True, slots=True)
class KernelCatalogEntry:
    """One Kernel revision flattened with its producing Agent and lineage context."""

    revision: KernelRevision
    revision_number: int
    parent_revision_number: int | None
    improvement_over_parent_percent: float | None
    campaign_id: CampaignId
    lineage_id: LineageId
    dsl: Dsl
    kernel_agent_revision_id: KernelAgentRevisionId
    kernel_agent_revision_number: int
    epoch_id: EpochId | None
    epoch_number: int | None
    attempt_id: AttemptId | None
    branch: BranchRole | None
    challenger_ordinal: int | None
    trajectory_ordinal: int | None
    attempt_ordinal: int | None
    accepted_as_branch_best: bool

    def __post_init__(self) -> None:
        if self.revision_number < 0:
            raise ValueError("Kernel revision number cannot be negative")
        if self.kernel_agent_revision_number < 0:
            raise ValueError("Kernel Agent revision number cannot be negative")
        if self.revision.parent_id is None:
            if self.parent_revision_number is not None:
                raise ValueError("root Kernel revision cannot have a parent revision number")
            if self.improvement_over_parent_percent is not None:
                raise ValueError("root Kernel revision cannot have a parent improvement")
        elif self.parent_revision_number is None:
            raise ValueError("child Kernel revision requires a parent revision number")
        attempt_fields = (
            self.epoch_id,
            self.epoch_number,
            self.attempt_id,
            self.branch,
            self.challenger_ordinal,
            self.trajectory_ordinal,
            self.attempt_ordinal,
        )
        if self.attempt_id is None:
            if any(value is not None for value in attempt_fields):
                raise ValueError("baseline Kernel catalog context cannot contain Attempt fields")
            if self.accepted_as_branch_best:
                raise ValueError("baseline Kernel cannot be an accepted Attempt candidate")
        elif any(value is None for value in attempt_fields):
            raise ValueError("Attempt Kernel catalog context is incomplete")


@dataclass(frozen=True, slots=True)
class KernelAgentCatalogEntry:
    """One lineage-local Kernel Agent revision and its evolution disposition."""

    revision: KernelAgentRevision
    revision_number: int
    parent_revision_number: int | None
    campaign_id: CampaignId
    lineage_id: LineageId
    introduced_epoch_id: EpochId | None
    introduced_epoch_number: int | None
    disposition: str
    active: bool

    def __post_init__(self) -> None:
        if self.revision_number < 0:
            raise ValueError("Kernel Agent revision number cannot be negative")
        if self.revision.parent_id is None:
            if self.parent_revision_number is not None:
                raise ValueError("root Kernel Agent cannot have a parent revision number")
        elif self.parent_revision_number is None:
            raise ValueError("child Kernel Agent requires a parent revision number")
        if (self.introduced_epoch_id is None) is not (self.introduced_epoch_number is None):
            raise ValueError("Kernel Agent introduction Epoch context is incomplete")
        if self.disposition not in {"baseline", "challenger", "promoted", "rejected", "failed"}:
            raise ValueError(f"unsupported Kernel Agent disposition: {self.disposition!r}")


@dataclass(frozen=True, slots=True)
class Lineage:
    """Independent evolution history for one DSL and evaluation contract."""

    id: LineageId
    campaign_id: CampaignId
    dsl: Dsl
    hardware_target: str
    active_kernel_agent_revision_id: KernelAgentRevisionId
    best_kernel_revision_id: KernelRevisionId
    evidence_checkpoint: ArtifactDigest
    challenger_count: int
    trajectories_per_branch: int
    attempts_per_trajectory: int
    next_epoch_number: int
    status: LineageStatus
    challenger_start_epoch: int = 1
    optimizer_model: str | None = None
    evolver_model: str | None = None

    def __post_init__(self) -> None:
        if self.challenger_count < 0:
            raise ValueError("a lineage cannot require a negative Challenger count")
        if self.challenger_start_epoch <= 0:
            raise ValueError("a lineage requires a positive Challenger start Epoch")
        if self.trajectories_per_branch <= 0:
            raise ValueError("a lineage requires at least one Trajectory per Branch")
        if self.attempts_per_trajectory <= 0:
            raise ValueError("a lineage requires a positive per-Trajectory Attempt budget")
        for role, model in (
            ("Optimizer", self.optimizer_model),
            ("Evolver", self.evolver_model),
        ):
            if model is not None and (not model.strip() or "\x00" in model):
                raise ValueError(f"Lineage {role} model is invalid")


@dataclass(frozen=True, slots=True)
class Epoch:
    """Durable state of one Active-versus-zero-or-more-Challengers competition."""

    id: EpochId
    lineage_id: LineageId
    number: int
    active_kernel_agent_revision_id: KernelAgentRevisionId
    challenger_kernel_agent_revision_ids: tuple[KernelAgentRevisionId, ...]
    starting_kernel_revision_id: KernelRevisionId
    evidence_checkpoint: ArtifactDigest
    challenger_count: int
    trajectories_per_branch: int
    attempts_per_trajectory: int
    status: EpochStatus
    winner_kernel_agent_revision_id: KernelAgentRevisionId | None
    best_kernel_revision_id: KernelRevisionId | None
    created_at: str
    completed_at: str | None

    def __post_init__(self) -> None:
        if self.challenger_count < 0:
            raise ValueError("an Epoch cannot require a negative Challenger count")
        if len(self.challenger_kernel_agent_revision_ids) > self.challenger_count:
            raise ValueError("an Epoch contains more Challengers than configured")
        if len(set(self.challenger_kernel_agent_revision_ids)) != len(
            self.challenger_kernel_agent_revision_ids
        ):
            raise ValueError("an Epoch cannot contain duplicate Challengers")
        if self.trajectories_per_branch <= 0 or self.attempts_per_trajectory <= 0:
            raise ValueError("an Epoch requires positive Trajectory and Attempt budgets")


@dataclass(frozen=True, slots=True)
class EpochChallenger:
    """Durable proposal provenance for one indexed Epoch Challenger."""

    epoch_id: EpochId
    challenger_ordinal: int
    kernel_agent_revision_id: KernelAgentRevisionId
    proposal_type: ChallengerProposalType
    base_revision_id: KernelAgentRevisionId
    evolution_trace_digest: ArtifactDigest

    def __post_init__(self) -> None:
        if self.challenger_ordinal <= 0:
            raise ValueError("Challenger ordinal must be positive")
        if (
            self.proposal_type is ChallengerProposalType.REUSE
            and self.base_revision_id != self.kernel_agent_revision_id
        ):
            raise ValueError("reused Challenger base must be the reused revision")


@dataclass(frozen=True, slots=True)
class Attempt:
    """Durable state of one optimization opportunity."""

    id: AttemptId
    epoch_id: EpochId
    branch: BranchRole
    challenger_ordinal: int
    trajectory_ordinal: int
    ordinal: int
    kernel_agent_revision_id: KernelAgentRevisionId
    input_kernel_revision_id: KernelRevisionId
    attempt_evidence_digest: ArtifactDigest
    output_kernel_revision_id: KernelRevisionId | None
    accepted_as_branch_best: bool
    status: AttemptStatus
    infrastructure_failures: int
    recovery_generation: int
    authority_started_at: str
    failure_reason: str | None
    created_at: str
    completed_at: str | None
    attempt_report_digest: ArtifactDigest | None = None
    attempt_report_status: AttemptReportStatus | None = None
    runtime_state_digest: ArtifactDigest | None = None
    input_runtime_state_digest: ArtifactDigest | None = None

    def __post_init__(self) -> None:
        if self.branch is BranchRole.ACTIVE and self.challenger_ordinal != 0:
            raise ValueError("an Active Attempt must use Challenger ordinal zero")
        if self.branch is BranchRole.CHALLENGER and self.challenger_ordinal <= 0:
            raise ValueError("a Challenger Attempt requires a positive Challenger ordinal")
        if self.trajectory_ordinal <= 0 or self.ordinal <= 0:
            raise ValueError("Attempt Trajectory and iteration ordinals must be positive")
        if self.recovery_generation < 0:
            raise ValueError("Attempt recovery generation cannot be negative")
        if (self.attempt_report_digest is None) is not (self.attempt_report_status is None):
            raise ValueError("Attempt report Digest and status must be stored together")


@dataclass(frozen=True, slots=True)
class EpochRecovery:
    """One operator-authorized, idempotent recovery of a failed epoch."""

    epoch_id: EpochId
    lineage_id: LineageId
    campaign_id: CampaignId
    recovery_key: str
    generation: int
    attempt_ids: tuple[AttemptId, ...]
    reason: str
    created_at: str


@dataclass(frozen=True, slots=True)
class CampaignTask:
    """One durable request to run a registered Campaign through an epoch target."""

    id: CampaignTaskId
    creation_key: str
    campaign_id: CampaignId
    target_epoch_number: int
    finalize: bool
    status: CampaignTaskStatus
    attempt_count: int
    lease_owner: str | None
    lease_expires_at: str | None
    last_error: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None

    def __post_init__(self) -> None:
        if self.target_epoch_number <= 0:
            raise ValueError("Campaign task target epoch must be positive")
        if self.attempt_count < 0:
            raise ValueError("Campaign task attempt count cannot be negative")


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """One ordered control-plane event without duplicated model content."""

    sequence: int
    kind: str
    aggregate_id: str
    payload: dict[str, object]
    created_at: str


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    """Bounded current control-plane counters derived from durable state."""

    latest_event_sequence: int
    event_counts: tuple[tuple[str, int], ...]
    campaign_task_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class AttemptSessionTrace:
    """One immutable Optimizer session artifact recorded for an Attempt run."""

    attempt_id: AttemptId
    run_ordinal: int
    artifact_digest: ArtifactDigest
    finish_reason: str
    token_budget: int
    token_usage: TokenUsage
    created_at: str

    def __post_init__(self) -> None:
        if self.run_ordinal <= 0:
            raise ValueError("an Attempt session trace requires a positive run ordinal")
        if not self.finish_reason:
            raise ValueError("an Attempt session trace requires a finish reason")
        if self.token_budget <= 0:
            raise ValueError("an Attempt session trace requires a positive token budget")

    @property
    def token_budget_exhausted(self) -> bool:
        """Return whether actual consumption reached or exceeded the session budget."""
        return self.token_usage.consumed >= self.token_budget


@dataclass(frozen=True, slots=True)
class BranchScore:
    """Metrics consumed by the pure Kernel Agent selection policy."""

    branch: BranchRole
    challenger_ordinal: int
    kernel_agent_revision_id: KernelAgentRevisionId
    best_latency_us: float
    first_best_attempt: int
    strict_improvements: int
    valid_candidates: int
    failed_candidates: int

    def __post_init__(self) -> None:
        if self.branch is BranchRole.ACTIVE and self.challenger_ordinal != 0:
            raise ValueError("the Active score must use Challenger ordinal zero")
        if self.branch is BranchRole.CHALLENGER and self.challenger_ordinal <= 0:
            raise ValueError("a Challenger score requires a positive Challenger ordinal")


@dataclass(frozen=True, slots=True)
class EpochSelection:
    """Atomic result committed when an epoch completes."""

    winner_kernel_agent_revision_id: KernelAgentRevisionId
    best_kernel_revision_id: KernelRevisionId
