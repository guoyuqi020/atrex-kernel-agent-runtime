"""Production composition for recoverable Campaign scheduler processes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from pathlib import Path

import anyio

from ..artifacts.local import ArtifactKind, LocalArtifactStore
from ..config import (
    CampaignRuntimeSettings,
    ComparisonSettings,
    EvaluateComparisonSettings,
    RuntimeSettings,
    SameAllocationAbbaComparisonSettings,
)
from ..controller.attempt_evidence import LocalAttemptEvidenceAssembler
from ..controller.campaign import CampaignScheduler
from ..controller.epoch import EpochController
from ..controller.evidence import LocalEvidenceAssembler
from ..controller.leases import RegistryLineageLeaseManager
from ..controller.projection import EvidenceArtifactProjector, EvidenceProjectionLimits
from ..domain.models import Attempt, Epoch, KernelMeasurementPurpose
from ..gateway.abba import AgateSameAllocationAbbaRunner, CommitPinnedAtrexBenchEvaluator
from ..gateway.agate import load_agate_sdk
from ..gateway.configuration import build_agate_connection
from ..gateway.contract import (
    RegistryKernelEvaluationContextResolver,
)
from ..gateway.control import (
    AttemptTimedWorkerGatewayAuthorityProvider,
    SqliteGatewayControl,
)
from ..gateway.control_models import GatewayOperation
from ..gateway.measurement import AgateKernelMeasurementRunner
from ..ports import (
    BuildChallengerRequest,
    BuildChallengerResult,
    EvolverRunner,
    KernelComparator,
    KernelMeasurementRunner,
    KernelPairMeasurementRunner,
)
from ..registry.sqlite import SqliteRegistry
from ..secrets import read_capability_signing_key
from ..selection import (
    OrdinaryEvaluateKernelComparator,
    SameAllocationAbbaKernelComparator,
)
from ..workers.core import (
    CoreOptimizerProcessConfig,
    CoreOptimizerSessionDriver,
)
from ..workers.evolution import (
    EVOLVER_WORKSPACE_RELATIVE_PATH,
    EvolutionProcessConfig,
    EvolutionSessionDriver,
    EvolutionWorkspaceAssembler,
    EvolverBundleRunner,
    SubprocessEvolutionSessionDriver,
)
from ..workers.evolver_bundle import GitEvolverBundleResolver
from ..workers.launcher import (
    BackendCredentialMounts,
    BwrapSandboxLauncher,
    CleanEnvironmentLauncher,
    WorkerLauncher,
)
from ..workers.optimizer import (
    OptimizerSessionConfig,
    OptimizerSessionDriver,
    SessionOptimizerRunner,
)
from ..workers.workspace import LocalAttemptWorkspaceAssembler
from .gateway import compose_authoritative_candidate_evaluator


class CampaignRuntime:
    """Own the resources used by one scheduler process."""

    def __init__(
        self,
        scheduler: CampaignScheduler,
        closers: tuple[Callable[[], None], ...],
    ) -> None:
        self.scheduler = scheduler
        self._closers = closers
        self._closed = False

    def close(self) -> None:
        """Close every owned durable connection once."""
        if self._closed:
            return
        self._closed = True
        for close in self._closers:
            close()

    def __enter__(self) -> CampaignRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class _LazyEvolverRunner:
    """Resolve Evolver credentials and Bundle only when an Epoch needs a Challenger."""

    def __init__(self, factory: Callable[[], EvolverRunner]) -> None:
        self._factory = factory
        self._runner: EvolverRunner | None = None
        self._lock = anyio.Lock()

    async def build_challenger(
        self,
        request: BuildChallengerRequest,
    ) -> BuildChallengerResult:
        runner = self._runner
        if runner is None:
            async with self._lock:
                runner = self._runner
                if runner is None:
                    runner = await anyio.to_thread.run_sync(self._factory)
                    self._runner = runner
        return await runner.build_challenger(request)


def build_campaign_runtime(
    settings: RuntimeSettings,
    environment: Mapping[str, str],
    *,
    optimizer_session_driver: OptimizerSessionDriver | None = None,
    evolution_session_driver: EvolutionSessionDriver | None = None,
    measurement_runner: KernelMeasurementRunner | None = None,
    pair_measurement_runner: KernelPairMeasurementRunner | None = None,
    attempt_finished: Callable[[Epoch, Attempt], None] | None = None,
) -> CampaignRuntime:
    """Assemble production Campaign workers or close partial resources on failure."""
    campaign = settings.campaign
    if campaign is None:
        raise ValueError("Runtime configuration does not define campaign worker settings")
    signing_key = read_capability_signing_key(
        environment,
        settings.gateway_proxy.capability_signing_key_env,
    )
    registry = SqliteRegistry(settings.storage.registry_database, require_fencing=True)
    control: SqliteGatewayControl | None = None
    try:
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        agate = settings.agate
        client, request_builder = load_agate_sdk(build_agate_connection(agate, environment))
        if measurement_runner is None and _requires_evaluate_measurement(campaign):
            measurement_runner = AgateKernelMeasurementRunner(
                client,
                request_builder,
                RegistryKernelEvaluationContextResolver(registry, artifacts),
                artifacts,
                registry,
                wait_timeout_s=agate.wait_timeout_s,
            )
        if pair_measurement_runner is None and _requires_same_allocation_abba(campaign):
            evaluator_settings = campaign.gate_policy.evaluator
            pair_measurement_runner = AgateSameAllocationAbbaRunner(
                client,
                RegistryKernelEvaluationContextResolver(registry, artifacts),
                artifacts,
                registry,
                CommitPinnedAtrexBenchEvaluator(
                    repository=evaluator_settings.repository,
                    commit=evaluator_settings.commit,
                    git_executable=evaluator_settings.git_executable,
                    fetch_timeout_seconds=evaluator_settings.fetch_timeout_seconds,
                    max_archive_bytes=evaluator_settings.max_archive_bytes,
                    max_bundle_files=evaluator_settings.max_bundle_files,
                    max_bundle_bytes=evaluator_settings.max_bundle_bytes,
                ),
                wait_timeout_s=agate.wait_timeout_s,
            )
        control = SqliteGatewayControl(
            settings.storage.gateway_database,
            registry,
            signing_key=signing_key,
        )
        worker_launcher = build_worker_launcher(settings, environment)
        optimizer_environment = campaign.optimizer.environment.resolve(environment)

        optimizer_config = OptimizerSessionConfig(environment=optimizer_environment)
        optimizer_sessions = optimizer_session_driver or CoreOptimizerSessionDriver(
            worker_launcher,
            build_core_process_config(campaign),
            artifacts,
        )
        authority = AttemptTimedWorkerGatewayAuthorityProvider(
            control,
            registry,
            campaign.gateway_proxy_url,
            operations=frozenset(
                {
                    *(GatewayOperation(operation) for operation in campaign.gateway_operations),
                    *((GatewayOperation.WIKI_QUERY,) if settings.gpu_wiki is not None else ()),
                }
            ),
            max_calls=campaign.gateway_max_calls,
            lifetime=timedelta(seconds=campaign.gateway_capability_lifetime_seconds),
        )
        optimizer = SessionOptimizerRunner(
            LocalAttemptWorkspaceAssembler(
                campaign.attempt_workspaces_root,
                registry,
                artifacts,
            ),
            optimizer_sessions,
            control,
            compose_authoritative_candidate_evaluator(
                settings,
                artifacts,
                registry,
                control,
                client,
                request_builder,
            ),
            authority,
            registry,
            registry,
            optimizer_config,
            independent_final_evaluation=False,
            wiki_enabled=settings.gpu_wiki is not None,
            worker_sessions=registry,
            kernel_trials=control,
            backend=campaign.optimizer.agent_backend,
        )

        deferred_environment = dict(environment)

        def build_evolver() -> EvolverRunner:
            evolution_config = build_evolution_process_config(
                campaign,
                artifacts,
                deferred_environment,
            )
            evolution_sessions = evolution_session_driver or SubprocessEvolutionSessionDriver(
                worker_launcher,
                evolution_config,
            )
            return EvolverBundleRunner(
                EvolutionWorkspaceAssembler(
                    campaign.evolution_workspaces_root,
                    artifacts,
                    evolver_bundle_digest=evolution_config.bundle_artifact_digest,
                ),
                evolution_sessions,
                artifacts,
                registry,
                kernel_agent_limits=settings.kernel_agent.bundle_limits(),
                max_output_manifest_bytes=campaign.evolver.max_output_manifest_bytes,
                worker_sessions=registry,
                backend=campaign.evolver.agent_backend,
            )

        evolver = _LazyEvolverRunner(build_evolver)
        evidence_projector = EvidenceArtifactProjector(
            artifacts,
            EvidenceProjectionLimits(
                max_trace_files=campaign.evidence.max_trace_files,
                max_trace_bytes=campaign.evidence.max_trace_bytes,
                max_trace_events=campaign.evidence.max_trace_events,
                max_projection_text_bytes=campaign.evidence.max_projection_text_bytes,
                max_diff_files=campaign.evidence.max_diff_files,
                max_diff_bytes=campaign.evidence.max_diff_bytes,
            ),
            redaction_patterns=campaign.evidence.redaction_patterns,
        )
        kernel_retention_comparator = _build_kernel_comparator(
            campaign.kernel_retention_comparison,
            measurement_runner,
            pair_measurement_runner,
            performance_timeout_seconds=campaign.gate_policy.performance_timeout_seconds,
            purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
        )
        agent_settings = campaign.agent_promotion_comparison
        agent_promotion_comparator = _build_kernel_comparator(
            agent_settings,
            measurement_runner,
            pair_measurement_runner,
            performance_timeout_seconds=campaign.gate_policy.performance_timeout_seconds,
            purpose=KernelMeasurementPurpose.AGENT_PROMOTION,
        )
        epochs = EpochController(
            registry,
            evolver,
            optimizer,
            LocalAttemptEvidenceAssembler(registry, artifacts, evidence_projector),
            kernel_retention_comparator=kernel_retention_comparator,
            agent_promotion_comparator=agent_promotion_comparator,
            max_infrastructure_retries=campaign.max_infrastructure_retries,
            max_parallel_branches=campaign.max_parallel_branches,
            agent_measurement_uncertainty_us=(
                agent_settings.measurement_uncertainty_us
                if isinstance(agent_settings, EvaluateComparisonSettings)
                else 0.0
            ),
            attempt_finished=attempt_finished,
        )
        scheduler = CampaignScheduler(
            registry,
            epochs,
            LocalEvidenceAssembler(
                registry,
                artifacts,
                evidence_projector,
                control,
            ),
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=campaign.fencing_lease_seconds,
                heartbeat_seconds=campaign.fencing_heartbeat_seconds,
            ),
            evolver_commit=campaign.evolver.commit,
        )
        return CampaignRuntime(scheduler, (control.close, registry.close))
    except BaseException:
        if control is not None:
            control.close()
        registry.close()
        raise


def build_evolution_process_config(
    campaign: CampaignRuntimeSettings,
    artifacts: LocalArtifactStore,
    environment: Mapping[str, str],
) -> EvolutionProcessConfig:
    """Resolve the exact commit-pinned Evolver process contract for runs or dev shells."""
    resolved = GitEvolverBundleResolver(
        artifacts,
        repository=campaign.evolver.repository,
        commit=campaign.evolver.commit,
        git_executable=campaign.evolver.git_executable,
        fetch_timeout_seconds=campaign.evolver.fetch_timeout_seconds,
        max_archive_bytes=campaign.evolver.max_archive_bytes,
        command_prefix=campaign.evolver.command_prefix,
        max_files=campaign.evolver.max_bundle_files,
        max_bytes=campaign.evolver.max_bundle_bytes,
    ).resolve()
    stored_bundle = artifacts.verify(resolved.artifact_digest)
    if stored_bundle.kind is not ArtifactKind.EVOLVER_BUNDLE:
        raise ValueError("resolved Evolver Bundle has the wrong Artifact kind")
    entrypoint = Path(resolved.command_argv[-1]).resolve()
    try:
        relative_entrypoint = entrypoint.relative_to(stored_bundle.payload_path.resolve())
    except ValueError as error:
        raise ValueError("resolved Evolver entrypoint escapes its sealed Bundle") from error
    workspace_entrypoint = EVOLVER_WORKSPACE_RELATIVE_PATH.joinpath(
        *relative_entrypoint.parts
    ).as_posix()
    return EvolutionProcessConfig(
        bundle_commit=resolved.commit,
        bundle_tree=resolved.tree,
        bundle_artifact_digest=resolved.artifact_digest,
        command_argv=(*resolved.command_argv[:-1], workspace_entrypoint),
        agent_backend=campaign.evolver.agent_backend,
        reasoning_effort=campaign.evolver.reasoning_effort,
        session_settings=campaign.evolver.session_settings,
        isolated_home_environment_keys=campaign.evolver.isolated_home_environment_keys,
        session_trace_relative_path=campaign.evolver.session_trace_relative_path,
        token_usage_report_relative_path=campaign.evolver.token_usage_report_relative_path,
        environment=campaign.evolver.environment.resolve(environment),
        timeout_seconds=campaign.evolver.timeout_seconds,
        terminate_grace_seconds=campaign.evolver.terminate_grace_seconds,
        max_diagnostic_bytes=campaign.evolver.max_diagnostic_bytes,
    )


def _requires_evaluate_measurement(campaign: CampaignRuntimeSettings) -> bool:
    """Return whether either trusted comparison policy needs fresh Agate samples."""
    return any(
        isinstance(comparison, EvaluateComparisonSettings)
        for comparison in (
            campaign.kernel_retention_comparison,
            campaign.agent_promotion_comparison,
        )
    )


def _requires_same_allocation_abba(campaign: CampaignRuntimeSettings) -> bool:
    """Return whether either comparison policy needs the paired dev runner."""
    return any(
        isinstance(comparison, SameAllocationAbbaComparisonSettings)
        for comparison in (
            campaign.kernel_retention_comparison,
            campaign.agent_promotion_comparison,
        )
    )


def _build_kernel_comparator(
    settings: ComparisonSettings,
    measurement_runner: KernelMeasurementRunner | None,
    pair_measurement_runner: KernelPairMeasurementRunner | None,
    *,
    performance_timeout_seconds: float,
    purpose: KernelMeasurementPurpose,
) -> KernelComparator:
    """Resolve one explicit comparison policy without silent method fallback."""
    if isinstance(settings, SameAllocationAbbaComparisonSettings):
        if pair_measurement_runner is None:
            raise ValueError(
                f"{purpose.value} selects same-allocation ABBA but no paired runner is configured"
            )
        return SameAllocationAbbaKernelComparator(
            pair_measurement_runner,
            repeats=settings.repeats,
            minimum_improvement_percent=settings.minimum_improvement_percent,
            per_run_timeout_seconds=performance_timeout_seconds,
            allocation_timeout_seconds=settings.allocation_timeout_seconds,
            shape_batch_size=settings.shape_batch_size,
            max_parallel_shape_batches=settings.max_parallel_shape_batches,
            purpose=purpose,
        )
    if measurement_runner is None:
        raise ValueError(
            f"{purpose.value} selects evaluate but no measurement runner is configured"
        )
    return OrdinaryEvaluateKernelComparator(
        measurement_runner,
        repeats=settings.repeats,
        measurement_uncertainty_us=settings.measurement_uncertainty_us,
        purpose=purpose,
    )


def build_worker_launcher(
    settings: RuntimeSettings,
    environment: Mapping[str, str],
) -> WorkerLauncher:
    """Build the explicit development launcher or strict production Sandbox."""
    campaign = settings.campaign
    if campaign is None:
        raise ValueError("Worker launcher requires Campaign runtime configuration")
    launch = campaign.launcher
    credentials = BackendCredentialMounts.from_environment(
        launch.backend_credentials,
        environment,
    )
    if launch.mode == "development":
        return CleanEnvironmentLauncher(launch.env_executable, credentials)
    sandbox = launch.sandbox
    if sandbox is None:
        raise ValueError("Sandbox launcher requires Sandbox settings")
    storage_hidden = (
        settings.storage.artifacts_root,
        settings.storage.registry_database.parent,
        settings.storage.gateway_database.parent,
        settings.storage.agate_jobs_database.parent,
    )
    sandbox = sandbox.model_copy(
        update={
            "hidden_host_paths": (*sandbox.hidden_host_paths, *storage_hidden),
        }
    )
    launcher = BwrapSandboxLauncher(
        launch.env_executable,
        sandbox,
        (
            campaign.attempt_workspaces_root,
            campaign.evolution_workspaces_root,
            campaign.problem_generalization_workspaces_root,
            campaign.lineage_bootstrap_workspaces_root,
        ),
        credentials,
    )
    launcher.check_host()
    return launcher


def build_core_process_config(
    campaign: CampaignRuntimeSettings,
    *,
    timeout_seconds: float | None = None,
) -> CoreOptimizerProcessConfig:
    """Share the Core process budget across Runtime-selected Core session phases."""
    worker = campaign.optimizer
    return CoreOptimizerProcessConfig(
        command_prefix=worker.command_prefix,
        agent_backend=worker.agent_backend,
        reasoning_effort=worker.reasoning_effort,
        session_settings=worker.session_settings,
        isolated_home_environment_keys=worker.isolated_home_environment_keys,
        session_trace_relative_path=worker.session_trace_relative_path,
        token_usage_report_relative_path=worker.token_usage_report_relative_path,
        max_attempt_report_bytes=worker.max_attempt_report_bytes,
        timeout_seconds=worker.timeout_seconds if timeout_seconds is None else timeout_seconds,
        terminate_grace_seconds=worker.terminate_grace_seconds,
        max_diagnostic_bytes=worker.max_diagnostic_bytes,
        max_session_tokens=worker.max_session_tokens,
        max_session_credits=worker.max_session_credits,
    )
