"""Immutable records and policies for trusted Gateway control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ..artifacts.local import JsonValue
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    EpochId,
    KernelAgentRevisionId,
    LineageId,
)
from ..domain.models import Dsl


class GatewayOperation(StrEnum):
    """Worker operations understood by the trusted Gateway proxy."""

    EVALUATE = "evaluate"
    SUBMIT = "submit"
    PROFILE = "profile"
    DEV = "dev"
    CHECK = "check"
    SOL = "sol"
    DISASSEMBLE = "disassemble"
    POLL = "poll"
    JOBS = "jobs"
    CANCEL = "cancel"
    ENV = "env"
    HEALTH = "health"
    CONFIG = "config"
    MEASUREMENTS = "measurements"
    KERNEL_TRIALS = "kernel_trials"
    KERNEL_TRIAL_READ = "kernel_trial_read"
    WIKI_QUERY = "wiki_query"


@dataclass(frozen=True, slots=True)
class GatewayCapability:
    """Bearer value injected into exactly one Attempt sandbox."""

    token: str
    attempt_id: AttemptId
    recovery_generation: int = 0


class BootstrapRunStatus(StrEnum):
    """Durable lifecycle of one physical framework-baseline execution."""

    ISSUED = "issued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GatewayEvaluationSource(StrEnum):
    """Authority that requested one immutable candidate evaluation."""

    AGENT = "agent"
    RUNTIME_FINAL = "runtime_final"


@dataclass(frozen=True, slots=True)
class GatewayEvaluationRecord:
    """One immutable Kernel snapshot and its exact Gateway evaluation outcome."""

    id: str
    attempt_id: AttemptId
    recovery_generation: int
    ordinal: int
    source: GatewayEvaluationSource
    idempotency_key: str
    candidate_artifact_digest: ArtifactDigest
    gateway_result_digest: ArtifactDigest
    correct: bool
    latency_us: float | None
    agate_job_id: str | None
    created_at: str

    def __post_init__(self) -> None:
        if not self.id.startswith("geval_"):
            raise ValueError("Gateway evaluation id must use the geval_ prefix")
        if self.recovery_generation < 0 or self.ordinal <= 0:
            raise ValueError("Gateway evaluation generation and ordinal are invalid")
        if not self.idempotency_key:
            raise ValueError("Gateway evaluation idempotency key cannot be empty")
        if self.correct and (self.latency_us is None or self.latency_us <= 0):
            raise ValueError("a correct Gateway evaluation requires positive latency")
        if not self.correct and self.latency_us is not None:
            raise ValueError("an incorrect Gateway evaluation cannot carry latency")


@dataclass(frozen=True, slots=True)
class GatewayMeasurementPoint:
    """One normalized safe measurement extracted from a Gateway result."""

    kind: GatewayOperation
    profile_level: str | None
    shape_id: str | None
    kernel_name: str | None
    metrics: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if self.kind not in {GatewayOperation.EVALUATE, GatewayOperation.PROFILE}:
            raise ValueError("historical measurement kind must be evaluate or profile")
        if not self.metrics:
            raise ValueError("historical measurement metrics cannot be empty")
        if any(not key or not isinstance(key, str) for key in self.metrics):
            raise ValueError("historical measurement metric names must be non-empty text")
        if any(isinstance(value, (dict, list)) for value in self.metrics.values()):
            raise ValueError("historical measurement metrics must be scalar JSON values")


@dataclass(frozen=True, slots=True)
class GatewayMeasurementRecord:
    """One immutable cross-Attempt measurement row retained by Runtime."""

    id: str
    attempt_id: AttemptId
    recovery_generation: int
    ordinal: int
    source_operation: GatewayOperation
    idempotency_key: str
    candidate_artifact_digest: ArtifactDigest
    gateway_result_digest: ArtifactDigest
    point: GatewayMeasurementPoint
    created_at: str

    def __post_init__(self) -> None:
        if not self.id.startswith("gmeasure_"):
            raise ValueError("Gateway measurement ID must use the gmeasure_ prefix")
        if self.recovery_generation < 0 or self.ordinal <= 0:
            raise ValueError("Gateway measurement generation and ordinal are invalid")
        if self.source_operation not in {
            GatewayOperation.EVALUATE,
            GatewayOperation.PROFILE,
        }:
            raise ValueError("Gateway measurement source must be evaluate or profile")


@dataclass(frozen=True, slots=True)
class GatewayKernelTrialObservation:
    """One Gateway operation performed against an exact experimental Kernel snapshot."""

    idempotency_key: str
    operation: GatewayOperation
    request_digest: ArtifactDigest
    result_artifact_digest: ArtifactDigest | None
    created_at: str


@dataclass(frozen=True, slots=True)
class GatewayKernelTrialAnnotation:
    """One Agent-authored experiment decision bound to an immutable Kernel snapshot."""

    sequence: int
    decision: str
    experiment: dict[str, JsonValue]
    recorded_at: str

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("Kernel Trial annotation sequence must be positive")
        if self.decision not in {"continue", "revert", "pivot"}:
            raise ValueError("Kernel Trial annotation decision is invalid")


@dataclass(frozen=True, slots=True)
class GatewayKernelTrialRecord:
    """One durable exact Kernel snapshot observed during an optimization Attempt."""

    id: str
    attempt_id: AttemptId
    recovery_generation: int
    ordinal: int
    candidate_artifact_digest: ArtifactDigest
    observations: tuple[GatewayKernelTrialObservation, ...]
    annotations: tuple[GatewayKernelTrialAnnotation, ...]
    created_at: str

    def __post_init__(self) -> None:
        if not self.id.startswith("gtrial_"):
            raise ValueError("Gateway Kernel Trial ID must use the gtrial_ prefix")
        if self.recovery_generation < 0 or self.ordinal <= 0:
            raise ValueError("Gateway Kernel Trial generation and ordinal are invalid")
        if not self.observations:
            raise ValueError("Gateway Kernel Trial requires at least one observation")

    @property
    def disposition(self) -> str:
        """Return the latest explicit decision, or observed when none was published."""
        return self.annotations[-1].decision if self.annotations else "observed"


@dataclass(frozen=True, slots=True)
class BootstrapRunOperationRecord:
    """One retained Gateway operation issued by an exact Bootstrap generation."""

    idempotency_key: str
    operation: GatewayOperation
    request_digest: ArtifactDigest
    result_artifact_digest: ArtifactDigest | None
    created_at: str


@dataclass(frozen=True, slots=True)
class BootstrapRunRecord:
    """Append-only-generation audit record for one Bootstrap Agent session."""

    attempt_id: AttemptId
    recovery_generation: int
    status: BootstrapRunStatus
    run_id: str | None
    workspace_path: str | None
    finish_reason: str | None
    failure_reason: str | None
    started_at: str
    completed_at: str | None
    session_trace_digest: ArtifactDigest | None
    token_budget: int | None
    uncached_input_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    output_tokens: int | None
    credits: float | None
    report_digest: ArtifactDigest | None
    candidate_digest: ArtifactDigest | None
    gateway_result_digest: ArtifactDigest | None
    operations: tuple[BootstrapRunOperationRecord, ...] = ()

    @property
    def total_tokens(self) -> int | None:
        values = (
            self.uncached_input_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.output_tokens,
        )
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    @property
    def usage_unit(self) -> str | None:
        if self.credits is not None:
            return "credits"
        return None if self.total_tokens is None else "provider_tokens"

    @property
    def consumed(self) -> float | int | None:
        return self.credits if self.credits is not None else self.total_tokens


@dataclass(frozen=True, slots=True)
class GatewayCapabilityPolicy:
    """Fixed operations and benchmark-call budget for one Attempt."""

    operations: frozenset[GatewayOperation]
    max_calls: int
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("Gateway capability requires at least one operation")
        if self.max_calls <= 0:
            raise ValueError("Gateway capability max_calls must be positive")
        if self.expires_at.tzinfo is None:
            raise ValueError("Gateway capability expiry must be timezone-aware")


@dataclass(frozen=True, slots=True)
class GatewayAuthorization:
    """Trusted same-process proof that one operation passed capability checks."""

    attempt_id: AttemptId
    operation: GatewayOperation
    idempotency_key: str
    request_digest: str
    recovery_generation: int


@dataclass(frozen=True, slots=True)
class BootstrapGatewaySubject:
    """Durable pre-Lineage identity and evaluation context for framework bring-up."""

    attempt_id: AttemptId
    campaign_id: CampaignId
    lineage_id: LineageId
    epoch_id: EpochId
    kernel_agent_revision_id: KernelAgentRevisionId
    operator: str
    hardware_target: str
    dsl: Dsl
    evaluation_contract_digest: ArtifactDigest
    input_kernel_digest: ArtifactDigest
    evidence_digest: ArtifactDigest
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.operator or not self.hardware_target:
            raise ValueError("bootstrap Gateway subject requires operator and hardware target")
        if self.created_at.tzinfo is None:
            raise ValueError("bootstrap Gateway subject creation time must be timezone-aware")
