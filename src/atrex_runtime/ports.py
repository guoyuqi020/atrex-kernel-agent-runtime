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
class BuildChallengerRequest:
    """Immutable input to one idempotent Evolver invocation."""

    parent_revision: KernelAgentRevision
    epoch_id: EpochId
    evidence_checkpoint: ArtifactDigest
    idempotency_key: str
    agent_catalog: tuple[KernelAgentCatalogEntry, ...] = ()
    kernel_catalog: tuple[KernelCatalogEntry, ...] = ()
    model: str | None = None


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


KernelAgentChallengerProposal = KernelAgentCandidateProposal | KernelAgentReuseProposal


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
