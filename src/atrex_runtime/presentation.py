"""Shared JSON projections for CLI and administration API read models."""

from __future__ import annotations

from collections.abc import Callable

from .artifacts.local import ArtifactKind, LocalArtifactStore
from .bootstrap import BootstrapResult, CampaignBootstrapResult
from .domain.ids import KernelAgentRevisionId, KernelRevisionId, LineageId
from .domain.models import (
    Attempt,
    AttemptReportStatus,
    AttemptStatus,
    BranchRole,
    Epoch,
    EpochChallenger,
    KernelAgentCatalogEntry,
    KernelCatalogEntry,
    KernelMeasurement,
    KernelMeasurementPurpose,
    Lineage,
    WorkerSession,
)
from .gateway.control_models import BootstrapRunRecord, GatewayEvaluationRecord
from .gateway.result_metrics import gateway_result_sol_percent
from .lineage_seed import LineageSeedResult
from .registry.base import Registry


def worker_session_value(session: WorkerSession) -> dict[str, object]:
    """Render one unified Worker lifecycle record without hiding raw-trace identity."""
    usage = session.token_usage
    return {
        "worker_session_id": session.id,
        "role": session.role.value,
        "subject_id": session.subject_id,
        "external_run_id": session.external_run_id,
        "campaign_id": session.campaign_id,
        "lineage_id": session.lineage_id,
        "epoch_id": session.epoch_id,
        "attempt_id": session.attempt_id,
        "recovery_generation": session.recovery_generation,
        "backend": session.backend,
        "model": session.model,
        "workspace_path": session.workspace_path,
        "status": session.status.value,
        "finish_reason": session.finish_reason,
        "session_trace_digest": session.trace_digest,
        "usage_unit": None if usage is None else usage.usage_unit,
        "usage_budget": session.token_budget,
        "token_budget": session.token_budget,
        "token_usage": (
            None
            if usage is None
            else {
                "uncached_input_tokens": usage.uncached_input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "total_tokens": usage.total_tokens,
                "credits": usage.credits,
                "consumed": usage.consumed,
                "usage_unit": usage.usage_unit,
            }
        ),
        "process_returncode": session.process_returncode,
        "error_type": session.error_type,
        "error_message": session.error_message,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
    }


def lineage_bootstrap_value(result: BootstrapResult) -> dict[str, object]:
    return {
        "campaign_id": result.campaign_id,
        "lineage_id": result.lineage_id,
        "bootstrap_attempt_id": result.bootstrap_attempt_id,
        "kernel_agent_revision_id": result.kernel_agent_revision_id,
        "kernel_agent_version": "agent-v0",
        "models": {"optimizer": result.optimizer_model, "evolver": result.evolver_model},
        "baseline_kernel_revision_id": result.baseline_kernel_revision_id,
        "baseline_kernel_version": "v0",
        "baseline_kernel_created_at": result.baseline_kernel_created_at,
        "baseline_kernel": {
            "kernel_revision_id": result.baseline_kernel_revision_id,
            "version": "v0",
            "created_at": result.baseline_kernel_created_at,
            "producer": {"kind": "bootstrap", "attempt_id": result.bootstrap_attempt_id},
        },
        "kernel_agent": {
            "kernel_agent_revision_id": result.kernel_agent_revision_id,
            "version": "agent-v0",
            "created_at": result.kernel_agent_created_at,
            "producer": {"kind": "bootstrap"},
            "optimizer_artifact": {
                "digest": result.optimizer_digest,
                "kind": ArtifactKind.KERNEL_AGENT,
                "referenced_at": result.kernel_agent_created_at,
            },
        },
        "evaluation_contract_digest": result.evaluation_contract_digest,
        "agent_problem_digest": result.agent_problem_digest,
        "initial_evidence_digest": result.initial_evidence_digest,
    }


def bootstrap_run_value(run: BootstrapRunRecord) -> dict[str, object]:
    return {
        "bootstrap_attempt_id": run.attempt_id,
        "recovery_generation": run.recovery_generation,
        "status": run.status.value,
        "run_id": run.run_id,
        "workspace_path": run.workspace_path,
        "finish_reason": run.finish_reason,
        "failure_reason": run.failure_reason,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "session_trace_digest": run.session_trace_digest,
        "token_budget": run.token_budget,
        "usage_unit": run.usage_unit,
        "usage_budget": run.token_budget,
        "token_usage": (
            None
            if run.consumed is None
            else {
                "uncached_input_tokens": run.uncached_input_tokens,
                "cache_read_tokens": run.cache_read_tokens,
                "cache_write_tokens": run.cache_write_tokens,
                "output_tokens": run.output_tokens,
                "total_tokens": run.total_tokens,
                "credits": run.credits,
                "consumed": run.consumed,
                "usage_unit": run.usage_unit,
            }
        ),
        "report_digest": run.report_digest,
        "candidate_digest": run.candidate_digest,
        "gateway_result_digest": run.gateway_result_digest,
        "operations": [
            {
                "idempotency_key": operation.idempotency_key,
                "operation": operation.operation.value,
                "request_digest": operation.request_digest,
                "result_artifact_digest": operation.result_artifact_digest,
                "created_at": operation.created_at,
            }
            for operation in run.operations
        ],
    }


def evaluation_value(evaluation: GatewayEvaluationRecord) -> dict[str, object]:
    label = f"g{evaluation.recovery_generation}-e{evaluation.ordinal}"
    return {
        "evaluation_id": evaluation.id,
        "evaluation_label": label,
        "attempt_id": evaluation.attempt_id,
        "recovery_generation": evaluation.recovery_generation,
        "ordinal": evaluation.ordinal,
        "source": evaluation.source.value,
        "idempotency_key": evaluation.idempotency_key,
        "kernel_artifact_digest": evaluation.kernel_artifact_digest,
        "candidate_artifact": {
            "digest": evaluation.kernel_artifact_digest,
            "kind": ArtifactKind.KERNEL,
            "referenced_at": evaluation.created_at,
        },
        "gateway_result_digest": evaluation.gateway_result_digest,
        "gateway_result_artifact": {
            "digest": evaluation.gateway_result_digest,
            "kind": ArtifactKind.GATEWAY_RESULT,
            "referenced_at": evaluation.created_at,
        },
        "correct": evaluation.correct,
        "latency_us": evaluation.latency_us,
        "agate_job_id": evaluation.agate_job_id,
        "created_at": evaluation.created_at,
    }


def kernel_trial_value(trial: object) -> dict[str, object]:
    """Render one durable experimental Kernel snapshot without promoting it to vN."""
    from .gateway.control_models import GatewayKernelTrialRecord

    if not isinstance(trial, GatewayKernelTrialRecord):
        raise TypeError("Kernel Trial presentation requires a GatewayKernelTrialRecord")
    return {
        "kernel_trial_id": trial.id,
        "attempt_id": trial.attempt_id,
        "recovery_generation": trial.recovery_generation,
        "ordinal": trial.ordinal,
        "trial_label": f"g{trial.recovery_generation}-t{trial.ordinal}",
        "kernel_artifact_digest": trial.kernel_artifact_digest,
        "candidate_artifact": {
            "digest": trial.kernel_artifact_digest,
            "kind": ArtifactKind.KERNEL,
            "referenced_at": trial.created_at,
        },
        "disposition": trial.disposition,
        "observations": [
            {
                "idempotency_key": item.idempotency_key,
                "operation": item.operation.value,
                "request_digest": item.request_digest,
                "gateway_result_digest": item.gateway_result_digest,
                "result_artifact_digest": item.result_artifact_digest,
                "created_at": item.created_at,
            }
            for item in trial.observations
        ],
        "annotations": [
            {
                "sequence": item.sequence,
                "disposition": item.disposition,
                "experiment": item.experiment,
                "recorded_at": item.recorded_at,
            }
            for item in trial.annotations
        ],
        "created_at": trial.created_at,
    }


def bootstrap_result_value(result: CampaignBootstrapResult) -> dict[str, object]:
    return {
        "campaign_id": result.campaign_id,
        "hardware_target": result.hardware_target,
        "agate_gpu": result.agate_gpu,
        "problem_generalization_model": result.problem_generalization_model,
        "evolver_commit": result.evolver_commit,
        "roofline": {"mode": result.roofline_mode, "detail": result.roofline_detail},
        "lineages": [lineage_bootstrap_value(lineage) for lineage in result.lineages],
    }


def lineage_seed_result_value(result: LineageSeedResult) -> dict[str, object]:
    """Render the immutable roots created for one seeded Lineage."""
    return {
        "campaign_id": result.campaign_id,
        "lineage_id": result.lineage_id,
        "dsl": result.dsl.value,
        "created_at": result.created_at,
        "kernel_agent": {
            "kernel_agent_revision_id": result.kernel_agent_revision_id,
            "version": "agent-v0",
            "artifact_digest": result.agent_artifact_digest,
            "source_provenance_digest": result.source_provenance_digest,
            "source_agent_revision_id": result.source_agent_revision_id,
        },
        "kernel": {
            "kernel_revision_id": result.kernel_revision_id,
            "version": "v0",
            "artifact_digest": result.kernel_artifact_digest,
            "gateway_result_digest": result.gateway_result_digest,
            "latency_us": result.latency_us,
            "source_kernel_revision_id": result.source_kernel_revision_id,
        },
        "initial_evidence_digest": result.evidence_checkpoint,
        "models": {"optimizer": result.optimizer_model, "evolver": result.evolver_model},
    }


def kernel_catalog_entry(registry: Registry, revision_id: KernelRevisionId) -> KernelCatalogEntry:
    lineage = registry.find_kernel_lineage(revision_id)
    for entry in registry.list_lineage_kernels(lineage.id):
        if entry.revision.id == revision_id:
            return entry
    raise RuntimeError("Kernel revision is absent from its resolved lineage catalog")


def agent_catalog_entry(
    registry: Registry,
    revision_id: KernelAgentRevisionId,
) -> KernelAgentCatalogEntry:
    lineage = registry.find_kernel_agent_lineage(revision_id)
    for entry in registry.list_lineage_agent_revisions(lineage.id):
        if entry.revision.id == revision_id:
            return entry
    raise RuntimeError("Kernel Agent revision is absent from its resolved lineage catalog")


def _attempt_disposition(attempt: Attempt) -> str:
    if attempt.status is AttemptStatus.RUNNING:
        return "running"
    if attempt.status is AttemptStatus.INFRASTRUCTURE_FAILED:
        return "infrastructure-failed"
    if attempt.output_kernel_revision_id is not None:
        return "retained" if attempt.accepted_as_branch_best else "rejected"
    if attempt.attempt_report_status is AttemptReportStatus.PIVOT:
        return "pivot"
    if attempt.attempt_report_status is AttemptReportStatus.BLOCKED:
        return "blocked"
    return "no-candidate"


def attempt_value(registry: Registry, attempt: Attempt) -> dict[str, object]:
    """Flatten one Attempt with lineage-local Kernel and Agent version labels."""
    epoch = registry.get_epoch(attempt.epoch_id)
    lineage = registry.get_lineage(epoch.lineage_id)
    kernel_numbers = {
        entry.revision.id: entry.revision_number
        for entry in registry.list_lineage_kernels(lineage.id)
    }
    agent_numbers = {
        entry.revision.id: entry.revision_number
        for entry in registry.list_lineage_agent_revisions(lineage.id)
    }
    input_number = kernel_numbers[attempt.input_kernel_revision_id]
    output_number = (
        None
        if attempt.output_kernel_revision_id is None
        else kernel_numbers[attempt.output_kernel_revision_id]
    )
    agent_number = agent_numbers[attempt.kernel_agent_revision_id]
    branch_label = (
        "active"
        if attempt.branch is BranchRole.ACTIVE
        else f"challenger-{attempt.challenger_ordinal}"
    )
    return {
        "attempt_id": attempt.id,
        "campaign_id": lineage.campaign_id,
        "lineage_id": lineage.id,
        "dsl": lineage.dsl,
        "epoch_id": epoch.id,
        "epoch_number": epoch.number,
        "branch": attempt.branch,
        "branch_label": branch_label,
        "challenger_ordinal": attempt.challenger_ordinal,
        "trajectory_ordinal": attempt.trajectory_ordinal,
        "attempt_ordinal": attempt.ordinal,
        "kernel_agent_revision_id": attempt.kernel_agent_revision_id,
        "kernel_agent_version": f"agent-v{agent_number}",
        "input_kernel_revision_id": attempt.input_kernel_revision_id,
        "input_kernel_version": f"v{input_number}",
        "output_kernel_revision_id": attempt.output_kernel_revision_id,
        "output_kernel_version": None if output_number is None else f"v{output_number}",
        "candidate_produced": attempt.output_kernel_revision_id is not None,
        "accepted_as_branch_best": attempt.accepted_as_branch_best,
        "status": attempt.status,
        "attempt_report_status": attempt.attempt_report_status,
        "disposition": _attempt_disposition(attempt),
        "failure_reason": attempt.failure_reason,
        "infrastructure_failures": attempt.infrastructure_failures,
        "recovery_generation": attempt.recovery_generation,
        "attempt_evidence_digest": attempt.attempt_evidence_digest,
        "attempt_report_digest": attempt.attempt_report_digest,
        "input_runtime_state_digest": attempt.input_runtime_state_digest,
        "runtime_state_digest": attempt.runtime_state_digest,
        "authority_started_at": attempt.authority_started_at,
        "created_at": attempt.created_at,
        "completed_at": attempt.completed_at,
    }


def lineage_attempt_values(registry: Registry, lineage_id: LineageId) -> list[dict[str, object]]:
    registry.get_lineage(lineage_id)
    return [
        attempt_value(registry, attempt)
        for epoch in registry.list_epochs(lineage_id)
        for attempt in registry.list_attempts(epoch.id)
    ]


def lineage_epoch_values(registry: Registry, lineage_id: LineageId) -> list[dict[str, object]]:
    lineage = registry.get_lineage(lineage_id)
    agent_versions = {
        entry.revision.id: f"agent-v{entry.revision_number}"
        for entry in registry.list_lineage_agent_revisions(lineage_id)
    }
    kernel_versions = {
        entry.revision.id: f"v{entry.revision_number}"
        for entry in registry.list_lineage_kernels(lineage_id)
    }

    def agent_version(revision_id: KernelAgentRevisionId) -> str:
        try:
            return agent_versions[revision_id]
        except KeyError as error:
            raise RuntimeError(f"Epoch Agent {revision_id} has no lineage-local version") from error

    def kernel_version(revision_id: KernelRevisionId) -> str:
        try:
            return kernel_versions[revision_id]
        except KeyError as error:
            raise RuntimeError(
                f"Epoch Kernel {revision_id} has no lineage-local version"
            ) from error

    return [
        _epoch_value(
            epoch,
            lineage,
            agent_version,
            kernel_version,
            registry.list_epoch_challengers(epoch.id),
        )
        for epoch in registry.list_epochs(lineage_id)
    ]


def _epoch_value(
    epoch: Epoch,
    lineage: Lineage,
    agent_version: Callable[[KernelAgentRevisionId], str],
    kernel_version: Callable[[KernelRevisionId], str],
    challenger_proposals: list[EpochChallenger],
) -> dict[str, object]:
    winner_id = epoch.winner_kernel_agent_revision_id
    winner_challenger_ordinal: int | None = None
    winner_branch: str | None = None
    decision: str | None = None
    if winner_id is not None:
        if winner_id == epoch.active_kernel_agent_revision_id:
            winner_branch = BranchRole.ACTIVE.value
            winner_challenger_ordinal = 0
            decision = "active_retained"
        else:
            try:
                winner_challenger_ordinal = (
                    epoch.challenger_kernel_agent_revision_ids.index(winner_id) + 1
                )
            except ValueError as error:
                raise RuntimeError(f"Epoch {epoch.id} winner is not a competing Agent") from error
            winner_branch = BranchRole.CHALLENGER.value
            decision = "challenger_promoted"
    return {
        "campaign_id": lineage.campaign_id,
        "lineage_id": epoch.lineage_id,
        "dsl": lineage.dsl,
        "epoch_id": epoch.id,
        "epoch_number": epoch.number,
        "status": epoch.status,
        "active_kernel_agent_revision_id": epoch.active_kernel_agent_revision_id,
        "active_agent_version": agent_version(epoch.active_kernel_agent_revision_id),
        "challenger_kernel_agent_revision_ids": epoch.challenger_kernel_agent_revision_ids,
        "challenger_agent_versions": [
            agent_version(revision_id) for revision_id in epoch.challenger_kernel_agent_revision_ids
        ],
        "challenger_proposals": [
            {
                "challenger_ordinal": item.challenger_ordinal,
                "kernel_agent_revision_id": item.kernel_agent_revision_id,
                "agent_version": agent_version(item.kernel_agent_revision_id),
                "proposal_type": item.proposal_type,
                "base_kernel_agent_revision_id": item.base_revision_id,
                "base_agent_version": agent_version(item.base_revision_id),
                "evolution_trace_digest": item.evolution_trace_digest,
            }
            for item in challenger_proposals
        ],
        "winner_kernel_agent_revision_id": winner_id,
        "winner_agent_version": None if winner_id is None else agent_version(winner_id),
        "winner_branch": winner_branch,
        "winner_challenger_ordinal": winner_challenger_ordinal,
        "decision": decision,
        "starting_kernel_revision_id": epoch.starting_kernel_revision_id,
        "starting_kernel_version": kernel_version(epoch.starting_kernel_revision_id),
        "best_kernel_revision_id": epoch.best_kernel_revision_id,
        "best_kernel_version": (
            None
            if epoch.best_kernel_revision_id is None
            else kernel_version(epoch.best_kernel_revision_id)
        ),
        "evidence_checkpoint": epoch.evidence_checkpoint,
        "challenger_count": epoch.challenger_count,
        "trajectories_per_branch": epoch.trajectories_per_branch,
        "attempts_per_trajectory": epoch.attempts_per_trajectory,
        "created_at": epoch.created_at,
        "completed_at": epoch.completed_at,
    }


def agent_revision_value(entry: KernelAgentCatalogEntry) -> dict[str, object]:
    revision = entry.revision
    return {
        "kernel_agent_revision_id": revision.id,
        "agent_version": f"agent-v{entry.revision_number}",
        "revision_number": entry.revision_number,
        "parent_kernel_agent_revision_id": revision.parent_id,
        "parent_agent_version": (
            None
            if entry.parent_revision_number is None
            else f"agent-v{entry.parent_revision_number}"
        ),
        "campaign_id": entry.campaign_id,
        "lineage_id": entry.lineage_id,
        "dsl": revision.dsl,
        "introduced_epoch_id": entry.introduced_epoch_id,
        "introduced_epoch_number": entry.introduced_epoch_number,
        "created_by": revision.created_by,
        "created_at": revision.created_at,
        "disposition": entry.disposition,
        "active": entry.active,
        "optimizer_artifact_digest": revision.optimizer_digest,
        "optimizer_artifact": {
            "digest": revision.optimizer_digest,
            "kind": ArtifactKind.KERNEL_AGENT,
            "referenced_at": revision.created_at,
        },
        "source_provenance_digest": revision.source_provenance_digest,
        "evolution_trace_digest": revision.evolution_trace_digest,
        "runtime_state_digest": revision.runtime_state_digest,
    }


def kernel_value(
    entry: KernelCatalogEntry,
    *,
    artifacts: LocalArtifactStore | None = None,
    include_measurements: bool = False,
    registry: Registry | None = None,
) -> dict[str, object]:
    revision = entry.revision
    value: dict[str, object] = {
        "kernel_revision_id": revision.id,
        "version": f"v{entry.revision_number}",
        "revision_number": entry.revision_number,
        "parent_kernel_revision_id": revision.parent_id,
        "parent_version": (
            None if entry.parent_revision_number is None else f"v{entry.parent_revision_number}"
        ),
        "kernel_agent_revision_id": entry.kernel_agent_revision_id,
        "kernel_agent_version": f"agent-v{entry.kernel_agent_revision_number}",
        "kernel_agent_revision_number": entry.kernel_agent_revision_number,
        "campaign_id": entry.campaign_id,
        "lineage_id": entry.lineage_id,
        "dsl": entry.dsl,
        "epoch_id": entry.epoch_id,
        "epoch_number": entry.epoch_number,
        "attempt_id": entry.attempt_id,
        "branch": entry.branch,
        "branch_label": (
            None
            if entry.branch is None
            else "active"
            if entry.branch is BranchRole.ACTIVE
            else f"challenger-{entry.challenger_ordinal}"
        ),
        "challenger_ordinal": entry.challenger_ordinal,
        "trajectory_ordinal": entry.trajectory_ordinal,
        "attempt_ordinal": entry.attempt_ordinal,
        "accepted_as_branch_best": entry.accepted_as_branch_best,
        "disposition": (
            "baseline"
            if entry.attempt_id is None
            else "retained"
            if entry.accepted_as_branch_best
            else "rejected"
        ),
        "correct": revision.evaluation.correct,
        "latency_us": revision.evaluation.latency_us,
        "sol_percent": (
            None
            if artifacts is None
            else gateway_result_sol_percent(
                artifacts,
                revision.evaluation.gateway_result_digest,
            )
        ),
        "improvement_over_parent_percent": entry.improvement_over_parent_percent,
        "kernel_artifact_digest": revision.artifact_digest,
        "kernel_artifact": {
            "digest": revision.artifact_digest,
            "kind": ArtifactKind.KERNEL,
            "referenced_at": revision.created_at,
        },
        "gateway_result_digest": revision.evaluation.gateway_result_digest,
        "gateway_result_artifact": {
            "digest": revision.evaluation.gateway_result_digest,
            "kind": ArtifactKind.GATEWAY_RESULT,
            "referenced_at": revision.created_at,
        },
        "created_at": revision.created_at,
    }
    if include_measurements:
        if registry is None:
            raise ValueError("Kernel measurements require a Registry")
        value["measurements"] = measurement_values(registry, entry)
    return value


def measurement_values(
    registry: Registry,
    entry: KernelCatalogEntry,
) -> list[dict[str, object]]:
    revision = entry.revision
    primary_purpose = (
        KernelMeasurementPurpose.FRAMEWORK_BASELINE
        if entry.attempt_id is None
        else KernelMeasurementPurpose.ATTEMPT_EVALUATION
    )
    primary = {
        "measurement_id": f"primary:{revision.id}",
        "purpose": primary_purpose,
        "repeat": 0,
        "correct": revision.evaluation.correct,
        "latency_us": revision.evaluation.latency_us,
        "gateway_result_digest": revision.evaluation.gateway_result_digest,
        "agate_job_id": None,
        "created_at": revision.created_at,
    }
    return [
        primary,
        *[
            _measurement_value(measurement)
            for measurement in registry.list_kernel_measurements(revision.id)
        ],
    ]


def _measurement_value(measurement: KernelMeasurement) -> dict[str, object]:
    return {
        "measurement_id": measurement.id,
        "purpose": measurement.purpose,
        "repeat": measurement.repeat,
        "correct": measurement.correct,
        "latency_us": measurement.latency_us,
        "gateway_result_digest": measurement.gateway_result_digest,
        "agate_job_id": measurement.agate_job_id,
        "created_at": measurement.created_at,
    }
