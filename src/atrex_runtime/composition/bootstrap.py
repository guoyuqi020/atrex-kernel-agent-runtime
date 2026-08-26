"""Trusted composition for Core phases used before normal Campaign scheduling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import cast

import anyio

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..bootstrap import CampaignBootstrapper, GeneratedLineageBaseline, RooflineMode
from ..config import RuntimeSettings
from ..domain.errors import InfrastructureError
from ..domain.ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    EpochId,
    KernelAgentRevisionId,
    LineageId,
    new_worker_session_id,
    parse_epoch_id,
)
from ..domain.models import Dsl, WorkerSession, WorkerSessionRole, WorkerSessionStatus
from ..gateway.agate import AgateClient, AgateRequestBuilder, load_agate_sdk
from ..gateway.configuration import build_agate_connection
from ..gateway.contract import (
    AgateEvaluationOptionsV1,
    RuntimeGateContractPolicy,
)
from ..gateway.control import SqliteGatewayControl
from ..gateway.control_models import (
    BootstrapGatewaySubject,
    BootstrapRunRecord,
    BootstrapRunStatus,
    GatewayCapabilityPolicy,
    GatewayEvaluationSource,
    GatewayOperation,
    gateway_kernel_trial_id,
)
from ..gateway.environment import AgateHardwareTargetResolver
from ..gateway.lineage_seed import AgateLineageSeedEvaluator
from ..gateway.production_policy import ProductionKernelPolicy
from ..kernel_agents import GitOptimizerBaseLoader, KernelAgentRevisionBuilder
from ..lineage_seed import LineageSeeder
from ..ports import AttemptCandidateResult, AuthoritativeCandidateEvaluator
from ..registry.base import Registry
from ..roofline import AtrexBenchRooflineBuilder
from ..workers import (
    CoreAgentProblemGenerator,
    CoreLineageBootstrapSessionDriver,
    CoreProblemGeneralizationSessionDriver,
    LineageBootstrapManifestV2,
    LineageBootstrapSessionConfig,
    LineageBootstrapWorkspaceAssembler,
    ProblemGeneralizationWorkspaceAssembler,
)
from ..workers.attempt_report import AttemptReportV12
from .campaign import (
    build_core_process_config,
    build_worker_launcher,
)
from .gateway import build_authoritative_candidate_evaluator


def build_optimizer_base_loader(
    settings: RuntimeSettings,
    artifacts: LocalArtifactStore,
) -> GitOptimizerBaseLoader | None:
    """Compose the optional trusted Git importer for the initial Agent Bundle."""
    source = settings.kernel_agent.base_source
    if source is None:
        return None
    return GitOptimizerBaseLoader(
        artifacts,
        KernelAgentRevisionBuilder(
            artifacts,
            limits=settings.kernel_agent.bundle_limits(),
        ),
        repository=source.repository,
        git_executable=source.git_executable,
        timeout_seconds=source.fetch_timeout_seconds,
        max_archive_bytes=source.max_archive_bytes,
        allowed_submodules=source.allowed_submodules,
    )


def build_campaign_bootstrapper(
    settings: RuntimeSettings,
    artifacts: LocalArtifactStore,
    registry: Registry,
    environment: Mapping[str, str],
    *,
    control: SqliteGatewayControl | None = None,
    finalizer: AuthoritativeCandidateEvaluator | None = None,
    agate_client: AgateClient | None = None,
    roofline_resolved: Callable[[RooflineMode, str | None], None] | None = None,
) -> CampaignBootstrapper:
    """Compose the one canonical Campaign bootstrap path for CLI and HTTP entrypoints."""
    if agate_client is None:
        agate_client, _request_builder = load_agate_sdk(
            build_agate_connection(settings.agate, environment)
        )
    return CampaignBootstrapper(
        registry,
        artifacts,
        base_loader=build_optimizer_base_loader(settings, artifacts),
        problem_generator=build_core_problem_generator(settings, artifacts, environment, registry),
        baseline_generator=(
            None
            if control is None
            else build_core_lineage_baseline_generator(
                settings,
                artifacts,
                registry,
                control,
                environment,
                finalizer=finalizer,
            )
        ),
        roofline_builder=build_roofline_builder(settings),
        roofline_resolved=roofline_resolved,
        hardware_target_resolver=AgateHardwareTargetResolver(agate_client),
        evolver_commit=(None if settings.campaign is None else settings.campaign.evolver.commit),
        gate_contract_policy=build_gate_contract_policy(settings),
        max_parallel_lineages=(
            1 if settings.campaign is None else settings.campaign.bootstrap_max_parallel_lineages
        ),
    )


def build_gate_contract_policy(
    settings: RuntimeSettings,
) -> RuntimeGateContractPolicy | None:
    campaign = settings.campaign
    if campaign is None:
        return None
    gate = campaign.gate_policy
    retention = gate.retention
    return RuntimeGateContractPolicy(
        options=AgateEvaluationOptionsV1(
            num_correctness_cases=retention.correctness_cases,
            bench_iters=retention.bench_iters,
            atol=gate.atol,
            rtol=gate.rtol,
            timeout_s=gate.evaluation_timeout_seconds,
        ),
        lock_clocks=gate.lock_clocks,
        atrex_bench_version=gate.evaluator.agate_package_version,
        runner_overrides={
            "warmup_iters": gate.warmup_iters,
            "candidate_timeout_s": gate.candidate_timeout_seconds,
            "perf_timeout_s": gate.performance_timeout_seconds,
        },
        production_gate=gate.production_gate,
    )


def build_lineage_seeder(
    settings: RuntimeSettings,
    artifacts: LocalArtifactStore,
    registry: Registry,
    client: AgateClient,
    request_builder: AgateRequestBuilder,
    *,
    production_policy: ProductionKernelPolicy | None = None,
) -> LineageSeeder:
    """Compose the shared trusted path for Artifact-rooted Lineage creation."""
    connection = settings.agate
    return LineageSeeder(
        registry,
        artifacts,
        KernelAgentRevisionBuilder(
            artifacts,
            limits=settings.kernel_agent.bundle_limits(),
        ),
        AgateLineageSeedEvaluator(
            client,
            request_builder,
            registry,
            artifacts,
            registry,
            wait_timeout_s=connection.wait_timeout_s,
            profile_without_roofline=True,
            production_policy=production_policy or ProductionKernelPolicy(),
        ),
        evolver_commit=(None if settings.campaign is None else settings.campaign.evolver.commit),
    )


def build_roofline_builder(
    settings: RuntimeSettings,
) -> AtrexBenchRooflineBuilder | None:
    """Return the optional deployment-approved analytical Roofline builder."""
    campaign = settings.campaign
    if campaign is None or campaign.roofline_builder is None:
        return None
    builder = campaign.roofline_builder
    return AtrexBenchRooflineBuilder(
        repository=builder.repository,
        commit=builder.commit,
        git_executable=builder.git_executable,
        python_executable=builder.python_executable,
        fetch_timeout_seconds=builder.fetch_timeout_seconds,
        execution_timeout_seconds=builder.execution_timeout_seconds,
        max_archive_bytes=builder.max_archive_bytes,
        max_output_bytes=builder.max_output_bytes,
        sku_by_hardware_target=builder.sku_by_hardware_target,
    )


def build_core_problem_generator(
    settings: RuntimeSettings,
    artifacts: LocalArtifactStore,
    environment: Mapping[str, str],
    registry: Registry,
) -> CoreAgentProblemGenerator | None:
    """Return the configured private-input Core phase."""
    campaign = settings.campaign
    if campaign is None:
        return None
    return CoreAgentProblemGenerator(
        ProblemGeneralizationWorkspaceAssembler(
            campaign.problem_generalization_workspaces_root,
            artifacts,
        ),
        CoreProblemGeneralizationSessionDriver(
            build_worker_launcher(settings, environment),
            build_core_process_config(campaign),
            artifacts,
            max_problem_bytes=settings.kernel_agent.max_agent_problem_bytes,
        ),
        campaign.optimizer.environment.resolve(environment),
        worker_sessions=registry,
        backend=campaign.optimizer.agent_backend,
    )


@dataclass(frozen=True, slots=True)
class _BaselineFinalizationCheckpoint:
    """One completed Agent nomination that still needs Runtime finalization."""

    recovery_generation: int
    workspace_path: str | None
    session_trace_digest: ArtifactDigest
    report_digest: ArtifactDigest
    candidate_digest: ArtifactDigest
    nominated_gateway_result_digest: ArtifactDigest


class CoreLineageBaselineGenerator:
    """Issue pre-Lineage authority, run Core, and accept only its Gateway outcome."""

    def __init__(
        self,
        workspaces: LineageBootstrapWorkspaceAssembler,
        sessions: CoreLineageBootstrapSessionDriver,
        control: SqliteGatewayControl,
        registry: Registry,
        finalizer: AuthoritativeCandidateEvaluator,
        artifacts: LocalArtifactStore,
        *,
        gateway_endpoint: str,
        operations: frozenset[GatewayOperation],
        max_calls: int,
        capability_lifetime: timedelta,
        environment: tuple[tuple[str, str], ...],
        wiki_enabled: bool,
        backend: str | None = None,
        max_infrastructure_retries: int = 0,
    ) -> None:
        if max_infrastructure_retries < 0:
            raise ValueError("Bootstrap infrastructure retries cannot be negative")
        self._workspaces = workspaces
        self._sessions = sessions
        self._control = control
        self._registry = registry
        self._finalizer = finalizer
        self._artifacts = artifacts
        self._gateway_endpoint = gateway_endpoint
        self._operations = operations
        self._max_calls = max_calls
        self._capability_lifetime = capability_lifetime
        self._environment = environment
        self._wiki_enabled = wiki_enabled
        self._backend = backend
        self._max_infrastructure_retries = max_infrastructure_retries

    def generate(
        self,
        *,
        bootstrap_attempt_id: AttemptId,
        campaign_id: CampaignId,
        lineage_id: LineageId,
        kernel_agent_revision_id: KernelAgentRevisionId,
        optimizer_digest: ArtifactDigest,
        input_kernel_digest: ArtifactDigest,
        evaluation_contract_digest: ArtifactDigest,
        agent_problem_digest: ArtifactDigest,
        evidence_digest: ArtifactDigest,
        dsl: Dsl,
        operator: str,
        hardware_target: str,
        model: str | None = None,
    ) -> GeneratedLineageBaseline:
        """Retry transient Bootstrap infrastructure failures with fresh authority."""
        failures = 0
        while True:
            try:
                return self._generate_once(
                    bootstrap_attempt_id=bootstrap_attempt_id,
                    campaign_id=campaign_id,
                    lineage_id=lineage_id,
                    kernel_agent_revision_id=kernel_agent_revision_id,
                    optimizer_digest=optimizer_digest,
                    input_kernel_digest=input_kernel_digest,
                    evaluation_contract_digest=evaluation_contract_digest,
                    agent_problem_digest=agent_problem_digest,
                    evidence_digest=evidence_digest,
                    dsl=dsl,
                    operator=operator,
                    hardware_target=hardware_target,
                    model=model,
                )
            except InfrastructureError as error:
                failures += 1
                if failures > self._max_infrastructure_retries:
                    if self._max_infrastructure_retries:
                        error.add_note(
                            "Bootstrap infrastructure retry budget exhausted: "
                            f"failures={failures}, "
                            f"max_retries={self._max_infrastructure_retries}"
                        )
                    raise
                failed_generation = self._control.current_generation(bootstrap_attempt_id)
                self._registry.record_runtime_event(
                    "bootstrap.lineage_baseline_retrying",
                    bootstrap_attempt_id,
                    {
                        "campaign_id": campaign_id,
                        "lineage_id": lineage_id,
                        "kernel_agent_revision_id": kernel_agent_revision_id,
                        "dsl": dsl.value,
                        "failed_generation": failed_generation,
                        "next_generation": failed_generation + 1,
                        "retry_number": failures,
                        "max_infrastructure_retries": self._max_infrastructure_retries,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

    def _generate_once(
        self,
        *,
        bootstrap_attempt_id: AttemptId,
        campaign_id: CampaignId,
        lineage_id: LineageId,
        kernel_agent_revision_id: KernelAgentRevisionId,
        optimizer_digest: ArtifactDigest,
        input_kernel_digest: ArtifactDigest,
        evaluation_contract_digest: ArtifactDigest,
        agent_problem_digest: ArtifactDigest,
        evidence_digest: ArtifactDigest,
        dsl: Dsl,
        operator: str,
        hardware_target: str,
        model: str | None = None,
    ) -> GeneratedLineageBaseline:
        try:
            existing_subject = self._control.get_bootstrap_subject(bootstrap_attempt_id)
        except KeyError:
            created_at = datetime.now(UTC)
        else:
            created_at = existing_subject.created_at
        subject = BootstrapGatewaySubject(
            bootstrap_attempt_id,
            campaign_id,
            lineage_id,
            self._bootstrap_epoch_id(bootstrap_attempt_id),
            kernel_agent_revision_id,
            operator,
            hardware_target,
            dsl,
            evaluation_contract_digest,
            input_kernel_digest,
            evidence_digest,
            created_at,
        )
        capability = self._control.issue_bootstrap(
            subject,
            GatewayCapabilityPolicy(
                self._operations,
                self._max_calls,
                datetime.now(UTC) + self._capability_lifetime,
            ),
        )
        recovered = self._control.get_committed_outcome(bootstrap_attempt_id)
        if recovered is not None:
            self._control.revoke(bootstrap_attempt_id)
            completed = next(
                (
                    run
                    for run in reversed(self._control.list_bootstrap_runs(bootstrap_attempt_id))
                    if run.status is BootstrapRunStatus.COMPLETED
                    and run.report_digest is not None
                    and run.session_trace_digest is not None
                ),
                None,
            )
            if completed is None:
                raise RuntimeError(
                    "Committed Bootstrap outcome has no completed Report and Session Trace"
                )
            validated = self._validated_outcome(recovered, bootstrap_attempt_id)
            return GeneratedLineageBaseline(
                validated.artifact_digest,
                validated.gateway_result_digest,
                cast(float, validated.latency_us),
                cast(ArtifactDigest, completed.report_digest),
                cast(ArtifactDigest, completed.session_trace_digest),
            )
        result = None
        outcome = None
        terminal_reason = "infrastructure-error"
        worker_session_id = None
        worker_session_terminal = False
        checkpoint = None
        try:
            self._reconcile_superseded_worker_sessions(
                bootstrap_attempt_id,
                before_generation=capability.recovery_generation,
            )
            checkpoint = self._recover_finalization_checkpoint(
                bootstrap_attempt_id,
                before_generation=capability.recovery_generation,
            )
            if checkpoint is None:
                prepared = self._workspaces.prepare(
                    LineageBootstrapManifestV2(
                        bootstrap_attempt_id=bootstrap_attempt_id,
                        lineage_id=lineage_id,
                        kernel_agent_revision_id=kernel_agent_revision_id,
                        input_kernel_digest=input_kernel_digest,
                        optimizer_digest=optimizer_digest,
                        evaluation_contract_digest=evaluation_contract_digest,
                        agent_problem_digest=agent_problem_digest,
                        dsl=dsl,
                        operator=operator,
                        hardware_target=hardware_target,
                    )
                )
                self._control.begin_bootstrap_run(
                    bootstrap_attempt_id,
                    capability.recovery_generation,
                    run_id=prepared.root.name,
                    workspace_path=str(prepared.root),
                )
                worker_session_id = new_worker_session_id()
                self._registry.start_worker_session(
                    WorkerSession(
                        id=worker_session_id,
                        role=WorkerSessionRole.FRAMEWORK_BASELINE,
                        subject_id=str(bootstrap_attempt_id),
                        external_run_id=prepared.root.name,
                        workspace_path=str(prepared.root),
                        status=WorkerSessionStatus.RUNNING,
                        started_at=datetime.now(UTC).isoformat(),
                        campaign_id=campaign_id,
                        lineage_id=lineage_id,
                        epoch_id=subject.epoch_id,
                        attempt_id=bootstrap_attempt_id,
                        recovery_generation=capability.recovery_generation,
                        backend=self._backend,
                        model=model,
                    )
                )
                result = self._sessions.run(
                    prepared,
                    LineageBootstrapSessionConfig(
                        environment=self._environment,
                        gateway_endpoint=self._gateway_endpoint,
                        gateway_capability=capability.token,
                        model=model,
                        wiki_endpoint=self._gateway_endpoint if self._wiki_enabled else None,
                        wiki_capability=capability.token if self._wiki_enabled else None,
                    ),
                )
                if result.finish_reason.startswith("process-exit-"):
                    terminal_reason = result.finish_reason
                    raise InfrastructureError(
                        f"Core lineage baseline {result.finish_reason}"
                    )
                if result.finish_reason != "completed":
                    terminal_reason = result.finish_reason
                    raise RuntimeError(f"Core lineage baseline {result.finish_reason}")
                if result.report_error is not None:
                    terminal_reason = "invalid-report"
                    raise ValueError(f"invalid Core lineage baseline report: {result.report_error}")
                if result.report is None or result.report_digest is None:
                    terminal_reason = "missing-report"
                    raise ValueError("Core lineage baseline did not publish its terminal report")
                if result.report.status == "blocked":
                    terminal_reason = "blocked"
                    raise RuntimeError(f"Core lineage baseline blocked: {result.report.blocker}")
                if result.report.status != "candidate_ready":
                    terminal_reason = "invalid-report-status"
                    raise ValueError(
                        "Bootstrap Attempt must nominate a candidate or report blocked"
                    )
                if result.kernel_artifact_digest is None:
                    terminal_reason = "missing-candidate"
                    raise ValueError("Core lineage baseline did not submit a candidate Kernel")
                nominated_result_digest, nominated_generation = self._nominated_evaluation(
                    result.report,
                    bootstrap_attempt_id,
                    result.kernel_artifact_digest,
                    capability.recovery_generation,
                )
                terminal_reason = "invalid-report-references"
                self._control.record_kernel_trial_annotations(
                    bootstrap_attempt_id,
                    tuple(
                        experiment.model_dump(mode="json")
                        for experiment in result.report.experiments
                    ),
                    profile_supporting_results=(
                        ()
                        if result.report.profile_evidence is None
                        else tuple(
                            item.model_dump(mode="json")
                            for item in result.report.profile_evidence.supporting_results
                        )
                    ),
                    recovery_generation=capability.recovery_generation,
                    allow_baseline=True,
                )
                self._registry.finish_worker_session(
                    worker_session_id,
                    status=WorkerSessionStatus.COMPLETED,
                    finish_reason="completed",
                    trace_digest=result.session_trace_digest,
                    token_budget=result.token_budget,
                    token_usage=result.token_usage,
                )
                worker_session_terminal = True
                candidate_digest = result.kernel_artifact_digest
            else:
                self._control.begin_bootstrap_run(
                    bootstrap_attempt_id,
                    capability.recovery_generation,
                    run_id=f"finalization-only-from-generation-{checkpoint.recovery_generation}",
                    workspace_path=(
                        checkpoint.workspace_path or f"artifact:{checkpoint.session_trace_digest}"
                    ),
                )
                candidate_digest = checkpoint.candidate_digest
                nominated_result_digest = checkpoint.nominated_gateway_result_digest
                nominated_generation = checkpoint.recovery_generation
            terminal_reason = "finalization-error"
            outcome = anyio.run(
                partial(
                    self._finalizer.finalize,
                    bootstrap_attempt_id,
                    candidate_digest,
                    nominated_gateway_result_digest=nominated_result_digest,
                    nominated_recovery_generation=nominated_generation,
                )
            )
            self._validated_outcome(outcome, bootstrap_attempt_id)
        except BaseException as error:
            if worker_session_id is not None and not worker_session_terminal:
                timed_out = "wall-time limit" in str(error)
                self._registry.finish_worker_session(
                    worker_session_id,
                    status=(
                        WorkerSessionStatus.TIMED_OUT if timed_out else WorkerSessionStatus.FAILED
                    ),
                    finish_reason="timeout" if timed_out else terminal_reason,
                    trace_digest=(None if result is None else result.session_trace_digest),
                    token_budget=None if result is None else result.token_budget,
                    token_usage=None if result is None else result.token_usage,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            outcome = self._control.get_committed_outcome(bootstrap_attempt_id)
            trace_digest = (
                result.session_trace_digest
                if result is not None
                else None
                if checkpoint is None
                else checkpoint.session_trace_digest
            )
            report_digest = (
                result.report_digest
                if result is not None
                else None
                if checkpoint is None
                else checkpoint.report_digest
            )
            failed_candidate_digest: ArtifactDigest | None = (
                outcome.artifact_digest
                if outcome is not None
                else (
                    result.kernel_artifact_digest
                    if result is not None
                    else None
                    if checkpoint is None
                    else checkpoint.candidate_digest
                )
            )
            run = self._control.finish_bootstrap_run(
                bootstrap_attempt_id,
                capability.recovery_generation,
                status=BootstrapRunStatus.FAILED,
                finish_reason=terminal_reason,
                failure_reason=f"{type(error).__name__}: {error}",
                session_trace_digest=trace_digest,
                token_budget=None if result is None else result.token_budget,
                token_usage=None if result is None else result.token_usage,
                report_digest=report_digest,
                candidate_digest=failed_candidate_digest,
                gateway_result_digest=(None if outcome is None else outcome.gateway_result_digest),
            )
            self._registry.record_runtime_event(
                "bootstrap.lineage_baseline_failed",
                bootstrap_attempt_id,
                self._run_event_payload(
                    run,
                    campaign_id=campaign_id,
                    lineage_id=lineage_id,
                    kernel_agent_revision_id=kernel_agent_revision_id,
                ),
            )
            error.add_note(
                f"Bootstrap run: attempt_id={bootstrap_attempt_id}, "
                f"generation={run.recovery_generation}, run_id={run.run_id}"
            )
            raise
        else:
            if outcome is None:
                raise AssertionError("completed Bootstrap lost its authoritative outcome")
            if result is not None:
                completed_trace_digest = result.session_trace_digest
                completed_report_digest = result.report_digest
            elif checkpoint is not None:
                completed_trace_digest = checkpoint.session_trace_digest
                completed_report_digest = checkpoint.report_digest
            else:
                raise AssertionError("completed Bootstrap lost its finalization checkpoint")
            run = self._control.finish_bootstrap_run(
                bootstrap_attempt_id,
                capability.recovery_generation,
                status=BootstrapRunStatus.COMPLETED,
                finish_reason="completed",
                failure_reason=None,
                session_trace_digest=completed_trace_digest,
                token_budget=None if result is None else result.token_budget,
                token_usage=None if result is None else result.token_usage,
                report_digest=completed_report_digest,
                candidate_digest=outcome.artifact_digest,
                gateway_result_digest=outcome.gateway_result_digest,
            )
        finally:
            self._control.revoke(bootstrap_attempt_id)
        self._registry.record_runtime_event(
            "bootstrap.lineage_baseline_completed",
            bootstrap_attempt_id,
            {
                **self._run_event_payload(
                    run,
                    campaign_id=campaign_id,
                    lineage_id=lineage_id,
                    kernel_agent_revision_id=kernel_agent_revision_id,
                ),
                "latency_us": outcome.latency_us,
            },
        )
        return GeneratedLineageBaseline(
            outcome.artifact_digest,
            outcome.gateway_result_digest,
            cast(float, outcome.latency_us),
            cast(ArtifactDigest, run.report_digest),
            cast(ArtifactDigest, run.session_trace_digest),
        )

    def _reconcile_superseded_worker_sessions(
        self,
        attempt_id: AttemptId,
        *,
        before_generation: int,
    ) -> None:
        """Close Worker rows left running when an older generation was interrupted."""
        for session in self._registry.list_worker_sessions(attempt_id=attempt_id):
            if (
                session.status is WorkerSessionStatus.RUNNING
                and session.recovery_generation is not None
                and session.recovery_generation < before_generation
            ):
                self._registry.finish_worker_session(
                    session.id,
                    status=WorkerSessionStatus.FAILED,
                    finish_reason="superseded-by-retry",
                    error_type="InfrastructureError",
                    error_message=(
                        "Runtime restarted before the Bootstrap Worker session "
                        "recorded a terminal result"
                    ),
                )

    def _recover_finalization_checkpoint(
        self,
        attempt_id: AttemptId,
        *,
        before_generation: int,
    ) -> _BaselineFinalizationCheckpoint | None:
        """Recover a completed Agent nomination without running another Session."""
        for run in reversed(self._control.list_bootstrap_runs(attempt_id)):
            if run.recovery_generation >= before_generation:
                continue
            if run.status is not BootstrapRunStatus.FAILED or run.finish_reason not in {
                "infrastructure-error",
                "finalization-error",
            }:
                continue
            if run.report_digest is None or run.session_trace_digest is None:
                continue
            report_artifact = self._artifacts.verify(run.report_digest)
            if report_artifact.kind is not ArtifactKind.ATTEMPT_REPORT:
                raise ValueError("Bootstrap recovery report Artifact has the wrong kind")
            report = AttemptReportV12.model_validate_json(
                report_artifact.payload_path.joinpath("value.json").read_bytes()
            )
            if report.attempt_id != attempt_id:
                raise ValueError("Bootstrap recovery report belongs to another Attempt")
            if report.status != "candidate_ready" or run.candidate_digest is None:
                continue
            candidate = self._artifacts.verify(run.candidate_digest)
            if candidate.kind is not ArtifactKind.KERNEL:
                raise ValueError("Bootstrap recovery candidate Artifact has the wrong kind")
            result_digest, generation = self._nominated_evaluation(
                report,
                attempt_id,
                run.candidate_digest,
                None,
            )
            gateway_result = self._artifacts.verify(result_digest)
            if gateway_result.kind is not ArtifactKind.GATEWAY_RESULT:
                raise ValueError("Bootstrap recovery Gateway result Artifact has the wrong kind")
            return _BaselineFinalizationCheckpoint(
                recovery_generation=generation,
                workspace_path=run.workspace_path,
                session_trace_digest=run.session_trace_digest,
                report_digest=run.report_digest,
                candidate_digest=run.candidate_digest,
                nominated_gateway_result_digest=result_digest,
            )
        return None

    def _nominated_evaluation(
        self,
        report: AttemptReportV12,
        attempt_id: AttemptId,
        candidate_digest: ArtifactDigest,
        recovery_generation: int | None,
    ) -> tuple[ArtifactDigest, int]:
        """Resolve the correct Evaluate explicitly bound to the final Kernel by its Journal."""
        referenced_bindings: set[tuple[str, ArtifactDigest]] = set()
        for experiment in report.experiments:
            for subject in (experiment.before, experiment.after):
                if (
                    subject is not None
                    and subject.kernel_artifact_digest == candidate_digest
                ):
                    referenced_bindings.update(
                        (subject.kernel_trial_id, result)
                        for result in subject.gateway_result_digests
                    )
        evaluation = next(
            (
                item
                for item in reversed(self._control.list_evaluations(attempt_id))
                if (recovery_generation is None or item.recovery_generation == recovery_generation)
                and item.source is GatewayEvaluationSource.AGENT
                and item.kernel_artifact_digest == candidate_digest
                and (
                    gateway_kernel_trial_id(
                        attempt_id,
                        item.recovery_generation,
                        candidate_digest,
                    ),
                    item.gateway_result_digest,
                )
                in referenced_bindings
                and item.correct
            ),
            None,
        )
        if evaluation is None:
            raise ValueError(
                "Bootstrap candidate has no correct Agent Evaluate referenced by its Experiment "
                "Journal"
            )
        return evaluation.gateway_result_digest, evaluation.recovery_generation

    @staticmethod
    def _run_event_payload(
        run: BootstrapRunRecord,
        *,
        campaign_id: CampaignId,
        lineage_id: LineageId,
        kernel_agent_revision_id: KernelAgentRevisionId,
    ) -> dict[str, object]:
        return {
            "campaign_id": campaign_id,
            "lineage_id": lineage_id,
            "kernel_agent_revision_id": kernel_agent_revision_id,
            "recovery_generation": run.recovery_generation,
            "run_id": run.run_id,
            "workspace_path": run.workspace_path,
            "status": run.status.value,
            "finish_reason": run.finish_reason,
            "failure_reason": run.failure_reason,
            "kernel_artifact_digest": run.candidate_digest,
            "gateway_result_digest": run.gateway_result_digest,
            "baseline_report_artifact_digest": run.report_digest,
            "session_trace_digest": run.session_trace_digest,
            "token_budget": run.token_budget,
            "total_tokens": run.total_tokens,
        }

    @staticmethod
    def _bootstrap_epoch_id(attempt_id: AttemptId) -> EpochId:
        return parse_epoch_id(f"epoch_{str(attempt_id).removeprefix('attempt_')}")

    @staticmethod
    def _validated_outcome(
        outcome: AttemptCandidateResult, attempt_id: AttemptId
    ) -> AttemptCandidateResult:
        if not outcome.correct or outcome.latency_us is None:
            raise ValueError(f"Core lineage baseline is not correct: {attempt_id}")
        return outcome


def build_core_lineage_baseline_generator(
    settings: RuntimeSettings,
    artifacts: LocalArtifactStore,
    registry: Registry,
    control: SqliteGatewayControl,
    environment: Mapping[str, str],
    *,
    finalizer: AuthoritativeCandidateEvaluator | None = None,
) -> CoreLineageBaselineGenerator | None:
    """Compose the Runtime-controlled Core framework-baseline phase when configured."""
    campaign = settings.campaign
    if campaign is None:
        return None
    operations = frozenset(
        {
            *(GatewayOperation(value) for value in campaign.gateway_operations),
            *((GatewayOperation.WIKI_QUERY,) if settings.gpu_wiki is not None else ()),
        }
    )
    resolved_finalizer = finalizer or build_authoritative_candidate_evaluator(
        settings, artifacts, registry, control, environment
    )
    return CoreLineageBaselineGenerator(
        LineageBootstrapWorkspaceAssembler(
            campaign.lineage_bootstrap_workspaces_root,
            artifacts,
            attempt_workspaces_root=campaign.attempt_workspaces_root,
        ),
        CoreLineageBootstrapSessionDriver(
            build_worker_launcher(settings, environment),
            build_core_process_config(
                campaign,
                timeout_seconds=campaign.optimizer.bootstrap_timeout_seconds,
            ),
            artifacts,
        ),
        control,
        registry,
        resolved_finalizer,
        artifacts,
        gateway_endpoint=campaign.gateway_proxy_url,
        operations=operations,
        max_calls=campaign.gateway_max_calls,
        capability_lifetime=timedelta(seconds=campaign.gateway_capability_lifetime_seconds),
        environment=campaign.optimizer.environment.resolve(environment),
        wiki_enabled=settings.gpu_wiki is not None,
        backend=campaign.optimizer.agent_backend,
        max_infrastructure_retries=campaign.max_infrastructure_retries,
    )
