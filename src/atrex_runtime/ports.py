"""Replaceable capability interfaces used by the controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from .domain.ids import (
    ArtifactDigest,
    AttemptId,
    EpochId,
    KernelAgentRevisionId,
    KernelRevisionId,
    LineageId,
    WorkerSessionId,
)
from .domain.models import (
    AttemptReportStatus,
    AttemptSessionTrace,
    BranchRole,
    Dsl,
    KernelAgentCatalogEntry,
    KernelAgentRevision,
    KernelCatalogEntry,
    KernelMeasurement,
    KernelMeasurementPurpose,
    KernelRevision,
    TokenUsage,
    WorkerSession,
    WorkerSessionStatus,
)


@dataclass(frozen=True, slots=True)
class EvolutionReference:
    """One independent read-only Lineage made available to an Evolver."""

    name: str
    lineage_id: LineageId
    evidence_checkpoint: ArtifactDigest
    agent_catalog: tuple[KernelAgentCatalogEntry, ...] = ()
    kernel_catalog: tuple[KernelCatalogEntry, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.name
            or len(self.name) > 64
            or not self.name[0].islower()
            or not self.name[0].isalpha()
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in self.name
            )
        ):
            raise ValueError("Evolution reference name must be lowercase kebab-case")
        if any(entry.lineage_id != self.lineage_id for entry in self.agent_catalog):
            raise ValueError("Evolution reference Agent catalog disagrees with its Lineage")
        if any(entry.lineage_id != self.lineage_id for entry in self.kernel_catalog):
            raise ValueError("Evolution reference Kernel catalog disagrees with its Lineage")


@dataclass(frozen=True, slots=True)
class BuildChallengerRequest:
    """Immutable input to one idempotent Evolver invocation."""

    parent_revision: KernelAgentRevision
    epoch_id: EpochId
    evidence_checkpoint: ArtifactDigest
    idempotency_key: str
    agent_catalog: tuple[KernelAgentCatalogEntry, ...] = ()
    kernel_catalog: tuple[KernelCatalogEntry, ...] = ()
    model: str | None = None
    hardware_target: str = "unspecified"
    epoch_number: int = 2
    max_challengers: int = 1
    optimizer_attempt_budget: int = 6
    references: tuple[EvolutionReference, ...] = ()

    def __post_init__(self) -> None:
        if self.epoch_number <= 0:
            raise ValueError("Evolution Epoch number must be positive")
        if self.max_challengers < 0:
            raise ValueError("Evolution maximum Challengers cannot be negative")
        if self.optimizer_attempt_budget <= 0:
            raise ValueError("Evolution Optimizer Attempt budget must be positive")
        if not self.hardware_target.strip() or "\x00" in self.hardware_target:
            raise ValueError("Evolution hardware target is invalid")
        names = [reference.name for reference in self.references]
        if len(set(names)) != len(names):
            raise ValueError("Evolution references cannot reuse a name")
        lineage_ids = [reference.lineage_id for reference in self.references]
        if len(set(lineage_ids)) != len(lineage_ids):
            raise ValueError("Evolution references cannot reuse a Lineage")


@dataclass(frozen=True, slots=True)
class KernelAgentCandidate:
    """Candidate Optimizer source plus its per-Trajectory adaptive-state seed."""

    dsl: Dsl
    optimizer_digest: ArtifactDigest
    runtime_state_digest: ArtifactDigest | None = None


@dataclass(frozen=True, slots=True)
class KernelAgentCandidateProposal:
    """A new revision derived from the Active or one historical revision."""

    proposal_type: Literal["evolved", "evolve_from_history"]
    base_revision_id: KernelAgentRevisionId
    candidate: KernelAgentCandidate


@dataclass(frozen=True, slots=True)
class KernelAgentReuseProposal:
    """An existing historical revision selected for a fresh competition."""

    proposal_type: Literal["reuse"]
    candidate_revision_id: KernelAgentRevisionId


@dataclass(frozen=True, slots=True)
class KernelAgentNoChangeProposal:
    """Do not create any more Challengers for this Epoch."""

    proposal_type: Literal["no_change"]


KernelAgentChallengerProposal = (
    KernelAgentCandidateProposal | KernelAgentReuseProposal | KernelAgentNoChangeProposal
)


@dataclass(frozen=True, slots=True)
class BuildChallengerResult:
    """Validated repository Challenger and its sealed Evolution provenance."""

    proposal: KernelAgentChallengerProposal
    evolution_trace_digest: ArtifactDigest


class EvolverRunner(Protocol):
    """Execute a Kernel Agent revision's Evolver role."""

    async def build_challenger(self, request: BuildChallengerRequest) -> BuildChallengerResult:
        """Return one complete Optimizer repository for the next revision."""
        ...


@dataclass(frozen=True, slots=True)
class RunAgentWorkflowRequest:
    """Trusted Epoch context supplied to one executable Agent Workflow."""

    revision: KernelAgentRevision
    epoch_id: EpochId
    epoch_number: int
    max_challengers: int
    optimizer_attempt_budget: int


class AgentWorkflowOperationHandler(Protocol):
    """Execute one capability-bounded operation requested by Workflow code."""

    async def execute_workflow_operation(
        self,
        operation: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Execute one idempotent Runtime service and return its safe projection."""
        ...


class AgentWorkflowRunner(Protocol):
    """Run an immutable Agent Revision's executable Epoch orchestration code."""

    async def run(
        self,
        request: RunAgentWorkflowRequest,
        operations: AgentWorkflowOperationHandler,
    ) -> None:
        """Run Workflow code until it durably completes the Epoch."""
        ...


@dataclass(frozen=True, slots=True)
class BuildAttemptEvidenceRequest:
    """Identity and immutable epoch input for one branch-local Evidence snapshot."""

    attempt_id: AttemptId
    epoch_id: EpochId
    branch: BranchRole
    challenger_ordinal: int
    trajectory_ordinal: int
    ordinal: int
    epoch_evidence_checkpoint: ArtifactDigest


class AttemptEvidenceAssembler(Protocol):
    """Seal history visible to exactly one future branch Attempt."""

    def assemble(self, request: BuildAttemptEvidenceRequest) -> ArtifactDigest:
        """Return an immutable snapshot containing only earlier same-branch Attempts."""
        ...

    def validate(
        self,
        digest: ArtifactDigest,
        request: BuildAttemptEvidenceRequest,
    ) -> None:
        """Verify a persisted snapshot is bound to the recovered Attempt identity."""
        ...


@dataclass(frozen=True, slots=True)
class RunAttemptRequest:
    """Immutable input to one fresh Optimizer session."""

    attempt_id: AttemptId
    kernel_agent_revision_id: KernelAgentRevisionId
    input_kernel_revision_id: KernelRevisionId
    epoch_evidence_checkpoint: ArtifactDigest
    attempt_evidence_digest: ArtifactDigest
    dsl: Dsl
    model: str | None = None


@dataclass(frozen=True, slots=True)
class AttemptCandidateResult:
    """Authoritative Gateway outcome for one generated candidate."""

    artifact_digest: ArtifactDigest
    gateway_result_digest: ArtifactDigest
    correct: bool
    latency_us: float | None

    def __post_init__(self) -> None:
        if self.correct and (self.latency_us is None or self.latency_us <= 0):
            raise ValueError("a correct candidate requires a positive latency")
        if not self.correct and self.latency_us is not None:
            raise ValueError("an incorrect candidate cannot carry a comparable latency")


@dataclass(frozen=True, slots=True)
class RunAttemptResult:
    """Normalized outcome of one fresh Optimizer session."""

    candidate: AttemptCandidateResult | None = None
    failure_reason: str | None = None
    attempt_report_digest: ArtifactDigest | None = None
    attempt_report_status: AttemptReportStatus | None = None

    def __post_init__(self) -> None:
        if (self.attempt_report_digest is None) is not (self.attempt_report_status is None):
            raise ValueError("Attempt report Digest and status must be returned together")


class OptimizerRunner(Protocol):
    """Run fresh Optimizer sessions for immutable Kernel Agent revisions."""

    async def run_attempt(self, request: RunAttemptRequest) -> RunAttemptResult:
        """Execute one optimization opportunity and return its normalized result."""
        ...


@dataclass(frozen=True, slots=True)
class KernelComparisonResult:
    """Trusted decision for one candidate against its exact incumbent."""

    accepted: bool
    reason: str
    authoritative_candidate: AttemptCandidateResult | None = None


class KernelComparator(Protocol):
    """Decide Kernel retention outside the evolvable Agent sandbox."""

    async def compare(
        self,
        incumbent: KernelRevision,
        candidate: KernelRevision,
    ) -> KernelComparisonResult:
        """Return whether the candidate is a measured strict improvement."""
        ...


@dataclass(frozen=True, slots=True)
class KernelMeasurementRun:
    """One independently executed ordinary evaluation repetition."""

    repeat: int
    correct: bool
    latency_us: float | None
    gateway_result_digest: ArtifactDigest | None = None
    agate_job_id: str | None = None


class KernelMeasurementRunner(Protocol):
    """Execute one ordinary single-Seed evaluation repetition for a Kernel."""

    async def run(
        self,
        revision: KernelRevision,
        repeat: int,
        purpose: KernelMeasurementPurpose,
    ) -> KernelMeasurementRun:
        """Return the authoritative result for one requested repetition."""

    def aggregate(
        self,
        revision: KernelRevision,
        runs: tuple[KernelMeasurementRun, ...],
        purpose: KernelMeasurementPurpose,
    ) -> ArtifactDigest:
        """Seal one arithmetic-mean comparison aggregate referencing every raw run."""


@dataclass(frozen=True, slots=True)
class KernelPairMeasurementResult:
    """Complete paired measurements produced by one same-allocation policy run."""

    incumbent_runs: tuple[KernelMeasurementRun, ...]
    candidate_runs: tuple[KernelMeasurementRun, ...]
    gateway_result_digest: ArtifactDigest | None = None
    incumbent_latency_us: float | None = None
    candidate_latency_us: float | None = None


class KernelPairMeasurementRunner(Protocol):
    """Execute an interleaved incumbent/candidate schedule on shared allocations."""

    async def run_pair(
        self,
        incumbent: KernelRevision,
        candidate: KernelRevision,
        *,
        repeats: int,
        purpose: KernelMeasurementPurpose,
        per_run_timeout_seconds: float,
        allocation_timeout_seconds: float,
        shape_batch_size: int,
        max_parallel_shape_batches: int,
    ) -> KernelPairMeasurementResult:
        """Return exact paired repetitions after every shape batch completes."""


class AttemptOutcomeSource(Protocol):
    """Read Gateway-authoritative state recorded outside the worker sandbox."""

    async def get_outcome(self, attempt_id: AttemptId) -> AttemptCandidateResult | None:
        """Return the verified outcome, or ``None`` when no evaluation was committed."""
        ...


class AuthoritativeCandidateEvaluator(Protocol):
    """Independently re-evaluate one Agent-nominated Kernel outside its sandbox."""

    async def finalize(
        self,
        attempt_id: AttemptId,
        kernel_artifact_digest: ArtifactDigest,
        *,
        nominated_gateway_result_digest: ArtifactDigest | None = None,
        nominated_recovery_generation: int | None = None,
        independent_evaluate: bool = True,
    ) -> AttemptCandidateResult:
        """Return and durably commit the Runtime-final authoritative outcome."""
        ...


class AttemptSessionTraceRecorder(Protocol):
    """Append immutable Optimizer session artifacts to one durable Attempt."""

    def record_attempt_session_trace(
        self,
        attempt_id: AttemptId,
        artifact_digest: ArtifactDigest,
        finish_reason: str,
        token_budget: int,
        token_usage: TokenUsage,
    ) -> AttemptSessionTrace:
        """Append and return the next run-ordinal trace record."""
        ...

    def record_attempt_runtime_state(
        self,
        attempt_id: AttemptId,
        runtime_state_digest: ArtifactDigest,
    ) -> None:
        """Attach the immutable post-Session state checkpoint to an Attempt."""
        ...


class RuntimeEventRecorder(Protocol):
    """Append versioned control-plane telemetry to durable storage."""

    def record_runtime_event(
        self,
        kind: str,
        aggregate_id: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        """Record one event whose payload contains no secrets or model content."""
        ...


class WorkerSessionRecorder(Protocol):
    """Persist lifecycle state for every independently launched Worker process."""

    def start_worker_session(self, session: WorkerSession) -> WorkerSession:
        """Store the running record before process launch."""
        ...

    def finish_worker_session(
        self,
        session_id: WorkerSessionId,
        *,
        status: WorkerSessionStatus,
        finish_reason: str,
        trace_digest: ArtifactDigest | None = None,
        token_budget: int | None = None,
        token_usage: TokenUsage | None = None,
        process_returncode: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> WorkerSession:
        """Store exactly one terminal result."""
        ...


class KernelMeasurementJournal(RuntimeEventRecorder, Protocol):
    """Persist measurement facts and their correlated lifecycle events."""

    def record_kernel_measurement(self, measurement: KernelMeasurement) -> KernelMeasurement:
        """Store one immutable measurement idempotently."""
        ...

    def get_authoritative_abba_batch(self, task_digest: ArtifactDigest) -> ArtifactDigest | None:
        """Find a completed physical ABBA batch by its exact request identity."""
        ...

    def record_authoritative_abba_batch(
        self, task_digest: ArtifactDigest, result_digest: ArtifactDigest
    ) -> ArtifactDigest:
        """Bind a completed physical ABBA batch to its immutable result."""
        ...


@dataclass(frozen=True, slots=True)
class WorkerGatewayAuthority:
    """Attempt-scoped proxy endpoint and bearer capability for one worker."""

    endpoint: str
    capability: str


class WorkerGatewayAuthorityProvider(Protocol):
    """Issue Gateway authority for exactly one durable Attempt."""

    async def get_authority(self, request: RunAttemptRequest) -> WorkerGatewayAuthority:
        """Return authority whose capability is bound to ``request.attempt_id``."""
        ...
