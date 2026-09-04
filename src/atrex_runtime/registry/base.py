"""Registry interface consumed by deterministic controllers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from ..domain.ids import (
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
from ..domain.models import (
    Attempt,
    AttemptReportStatus,
    AttemptSessionTrace,
    BranchRole,
    Campaign,
    CampaignTask,
    Epoch,
    EpochChallenger,
    EpochRecovery,
    EpochSelection,
    EpochStatus,
    KernelAgentCatalogEntry,
    KernelAgentRevision,
    KernelCatalogEntry,
    KernelEvaluation,
    KernelMeasurement,
    KernelRevision,
    Lineage,
    RuntimeEvent,
    RuntimeMetrics,
    TokenUsage,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)


class Registry(Protocol):
    """Durable operations required by the Runtime control plane."""

    def close(self) -> None: ...

    def check_health(self) -> None: ...

    def list_referenced_artifact_digests(self) -> set[ArtifactDigest]: ...

    def start_worker_session(self, session: WorkerSession) -> WorkerSession: ...

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
    ) -> WorkerSession: ...

    def get_worker_session(self, session_id: WorkerSessionId) -> WorkerSession: ...

    def list_worker_sessions(
        self,
        *,
        campaign_id: CampaignId | None = None,
        lineage_id: LineageId | None = None,
        epoch_id: EpochId | None = None,
        attempt_id: AttemptId | None = None,
        subject_id: str | None = None,
        role: WorkerSessionRole | None = None,
        status: WorkerSessionStatus | None = None,
    ) -> list[WorkerSession]: ...

    def insert_campaign(self, campaign: Campaign) -> None: ...

    def get_campaign(self, campaign_id: CampaignId) -> Campaign: ...

    def ensure_campaign_evolver_commit(self, campaign_id: CampaignId, commit: str) -> Campaign: ...

    def list_campaign_lineages(self, campaign_id: CampaignId) -> list[Lineage]: ...

    def complete_campaign(self, campaign_id: CampaignId, target_epoch: int) -> Campaign: ...

    def cancel_campaign(self, campaign_id: CampaignId) -> Campaign: ...

    def enqueue_campaign_task(self, task: CampaignTask) -> CampaignTask: ...

    def get_campaign_task(self, task_id: CampaignTaskId) -> CampaignTask: ...

    def claim_campaign_task(
        self, owner: str, *, now: str, lease_expires_at: str
    ) -> CampaignTask | None: ...

    def renew_campaign_task(
        self, task_id: CampaignTaskId, owner: str, *, lease_expires_at: str
    ) -> bool: ...

    def complete_campaign_task(self, task_id: CampaignTaskId, owner: str) -> CampaignTask: ...

    def fail_campaign_task(
        self, task_id: CampaignTaskId, owner: str, *, error: str
    ) -> CampaignTask: ...

    def cancel_campaign_task(self, task_id: CampaignTaskId) -> CampaignTask: ...

    def requeue_campaign_task(self, task_id: CampaignTaskId) -> CampaignTask: ...

    def list_runtime_events(
        self,
        *,
        after_sequence: int,
        limit: int,
        kinds: tuple[str, ...] = (),
        correlation: Mapping[str, str] | None = None,
    ) -> list[RuntimeEvent]: ...

    def prune_runtime_events(self, *, before_sequence: int, limit: int) -> int: ...

    def summarize_runtime_metrics(self) -> RuntimeMetrics: ...

    def record_runtime_event(
        self,
        kind: str,
        aggregate_id: str,
        payload: Mapping[str, object] | None = None,
    ) -> None: ...

    def register_kernel_agent_revision(
        self, revision: KernelAgentRevision
    ) -> KernelAgentRevision: ...

    def get_kernel_agent_revision(
        self, revision_id: KernelAgentRevisionId
    ) -> KernelAgentRevision: ...

    def find_kernel_agent_revision_by_creation_key(
        self, creation_key: str
    ) -> KernelAgentRevision | None: ...

    def list_lineage_agent_revisions(
        self, lineage_id: LineageId
    ) -> list[KernelAgentCatalogEntry]: ...

    def list_campaign_agent_revisions(
        self, campaign_id: CampaignId
    ) -> list[KernelAgentCatalogEntry]: ...

    def find_kernel_agent_lineage(self, revision_id: KernelAgentRevisionId) -> Lineage: ...

    def register_kernel_revision(self, revision: KernelRevision) -> KernelRevision: ...

    def finalize_kernel_revision_evaluation(
        self,
        revision_id: KernelRevisionId,
        evaluation: KernelEvaluation,
    ) -> KernelRevision: ...

    def get_kernel_revision(self, revision_id: KernelRevisionId) -> KernelRevision: ...

    def find_kernel_revision_by_attempt(self, attempt_id: AttemptId) -> KernelRevision | None: ...

    def list_lineage_kernels(self, lineage_id: LineageId) -> list[KernelCatalogEntry]: ...

    def list_campaign_kernels(self, campaign_id: CampaignId) -> list[KernelCatalogEntry]: ...

    def record_kernel_measurement(self, measurement: KernelMeasurement) -> KernelMeasurement: ...

    def list_kernel_measurements(
        self, revision_id: KernelRevisionId
    ) -> list[KernelMeasurement]: ...

    def insert_lineage(self, lineage: Lineage) -> None: ...

    def get_lineage(self, lineage_id: LineageId) -> Lineage: ...

    def find_kernel_lineage(self, kernel_revision_id: KernelRevisionId) -> Lineage: ...

    def advance_lineage_evidence(
        self,
        lineage_id: LineageId,
        expected: ArtifactDigest,
        next_checkpoint: ArtifactDigest,
    ) -> None: ...

    def insert_epoch(self, epoch: Epoch) -> None: ...

    def get_epoch(self, epoch_id: EpochId) -> Epoch: ...

    def list_epochs(self, lineage_id: LineageId) -> list[Epoch]: ...

    def find_epoch(self, lineage_id: LineageId, number: int) -> Epoch | None: ...

    def find_open_epoch(self, lineage_id: LineageId) -> Epoch | None: ...

    def attach_challenger(
        self,
        challenger: EpochChallenger,
    ) -> None: ...

    def list_epoch_challengers(self, epoch_id: EpochId) -> list[EpochChallenger]: ...

    def transition_epoch(
        self, epoch_id: EpochId, expected: EpochStatus, next_status: EpochStatus
    ) -> None: ...

    def fail_epoch(self, epoch_id: EpochId, reason: str) -> None: ...

    def stop_epoch(self, epoch_id: EpochId, reason: str) -> Epoch: ...

    def resume_stopped_epoch(self, epoch_id: EpochId) -> Epoch: ...

    def recover_failed_epoch(
        self,
        epoch_id: EpochId,
        *,
        recovery_key: str,
        reason: str,
    ) -> EpochRecovery: ...

    def insert_attempt(self, attempt: Attempt) -> None: ...

    def get_attempt(self, attempt_id: AttemptId) -> Attempt: ...

    def find_attempt(
        self,
        epoch_id: EpochId,
        branch: BranchRole,
        challenger_ordinal: int,
        trajectory_ordinal: int,
        ordinal: int,
    ) -> Attempt | None: ...

    def list_attempts(self, epoch_id: EpochId) -> list[Attempt]: ...

    def record_attempt_session_trace(
        self,
        attempt_id: AttemptId,
        artifact_digest: ArtifactDigest,
        finish_reason: str,
        token_budget: int,
        token_usage: TokenUsage,
    ) -> AttemptSessionTrace: ...

    def list_attempt_session_traces(self, attempt_id: AttemptId) -> list[AttemptSessionTrace]: ...

    def record_attempt_input_runtime_state(
        self,
        attempt_id: AttemptId,
        runtime_state_digest: ArtifactDigest,
    ) -> None: ...

    def record_attempt_runtime_state(
        self,
        attempt_id: AttemptId,
        runtime_state_digest: ArtifactDigest,
    ) -> None: ...

    def record_attempt_report(
        self,
        attempt_id: AttemptId,
        artifact_digest: ArtifactDigest,
        status: AttemptReportStatus,
    ) -> None: ...

    def retry_attempt(self, attempt_id: AttemptId) -> None: ...

    def record_infrastructure_failure(self, attempt_id: AttemptId, reason: str) -> None: ...

    def complete_attempt(
        self,
        attempt_id: AttemptId,
        output_kernel_revision_id: KernelRevisionId | None,
        *,
        accepted_as_branch_best: bool,
        failure_reason: str | None,
    ) -> None: ...

    def complete_epoch(self, epoch_id: EpochId, selection: EpochSelection) -> None: ...
