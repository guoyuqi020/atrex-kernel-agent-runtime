"""Interactive, non-Agent shells for real Optimizer and frozen Evolver workspaces."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anyio

from .artifacts.local import ArtifactKind, LocalArtifactStore
from .controller.attempt_evidence import AttemptEvidenceMetadataV2, LocalAttemptEvidenceAssembler
from .controller.leases import RegistryLineageLeaseManager
from .controller.projection import EvidenceArtifactProjector, EvidenceProjectionLimits
from .domain.ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    EpochId,
    KernelAgentRevisionId,
    KernelRevisionId,
    LineageId,
    new_attempt_id,
    new_epoch_id,
)
from .domain.models import (
    Attempt,
    AttemptStatus,
    BranchRole,
    Dsl,
    Epoch,
    EpochStatus,
)
from .filesystem import make_tree_owner_writable
from .gateway.control import AttemptTimedWorkerGatewayAuthorityProvider, SqliteGatewayControl
from .gateway.control_models import (
    BootstrapGatewaySubject,
    GatewayCapabilityPolicy,
    GatewayOperation,
)
from .ports import BuildAttemptEvidenceRequest, BuildChallengerRequest, RunAttemptRequest
from .registry.sqlite import SqliteRegistry
from .serialization import write_canonical_json
from .workers.core import CoreOptimizerSessionDriver
from .workers.evidence_view import assemble_optimizer_evidence_view
from .workers.evolution import (
    EvolutionWorkspaceAssembler,
    SubprocessEvolutionSessionDriver,
)
from .workers.manifest import (
    ATTEMPT_MANIFEST_RELATIVE_PATH,
    ATTEMPT_WORKSPACE_LAYOUT,
    AttemptInputManifestV9,
    AttemptTaskContextV5,
)
from .workers.optimizer import OptimizerSessionConfig
from .workers.workspace import (
    LocalAttemptWorkspaceAssembler,
    PreparedAttempt,
    initialize_reusable_agent_state,
    remove_optimizer_state_seeds,
)


@dataclass(frozen=True, slots=True)
class DevShellResult:
    """Durable identity and retained workspace from one interactive shell."""

    attempt_id: AttemptId
    lineage_id: LineageId
    workspace: Path
    shell: Path
    returncode: int
    created_attempt: bool


@dataclass(frozen=True, slots=True)
class EvolverDevShellResult:
    """Selected Epoch identity and retained frozen Evolution workspace."""

    lineage_id: LineageId
    epoch_id: EpochId
    epoch_number: int
    epoch_status: EpochStatus
    parent_revision_id: KernelAgentRevisionId
    workspace: Path
    shell: Path
    returncode: int


@dataclass(frozen=True, slots=True)
class TemporaryOptimizerDevShellRequest:
    """Trusted inputs for one disposable Optimizer-compatible workspace."""

    attempt_id: AttemptId
    campaign_id: CampaignId
    lineage_id: LineageId
    epoch_id: EpochId
    kernel_agent_revision_id: KernelAgentRevisionId
    input_kernel_revision_id: KernelRevisionId
    optimizer_digest: ArtifactDigest
    input_kernel_digest: ArtifactDigest
    evaluation_contract_digest: ArtifactDigest
    agent_problem_digest: ArtifactDigest
    epoch_evidence_checkpoint: ArtifactDigest
    attempt_evidence_digest: ArtifactDigest
    dsl: Dsl
    operator: str
    hardware_target: str
    model: str | None = None


@dataclass(frozen=True, slots=True)
class TemporaryDevShellResult:
    """Identity of one shell whose workspace has already been destroyed."""

    attempt_id: AttemptId
    workspace: Path
    shell: Path
    returncode: int


@dataclass(frozen=True, slots=True)
class TemporaryEvolverDevShellResult:
    """Identity of one Evolver shell whose workspace has already been destroyed."""

    epoch_id: EpochId
    parent_revision_id: KernelAgentRevisionId
    workspace: Path
    shell: Path
    returncode: int


def create_empty_attempt_evidence(
    artifacts: LocalArtifactStore,
    *,
    epoch_id: EpochId,
    attempt_id: AttemptId,
    epoch_evidence_checkpoint: ArtifactDigest,
) -> ArtifactDigest:
    """Seal an empty first-Attempt Evidence snapshot without Registry state."""
    with tempfile.TemporaryDirectory(prefix="atrex-temporary-attempt-evidence-") as temporary:
        root = Path(temporary)
        for name in ("attempts", "traces", "diffs", "reports"):
            (root / name).mkdir(mode=0o700)
        metadata = AttemptEvidenceMetadataV2(
            epoch_id=epoch_id,
            attempt_id=attempt_id,
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=1,
            epoch_evidence_checkpoint=epoch_evidence_checkpoint,
            previous_attempt_ids=(),
        )
        write_canonical_json(
            root / "context.json",
            metadata.model_dump(mode="json"),
        )
        write_canonical_json(
            root / "lessons.json",
            {"schema_version": 2, "annotations": []},
        )
        return artifacts.put_directory(root, ArtifactKind.ATTEMPT_EVIDENCE)


ShellRunner = Callable[[tuple[str, ...], Path, Mapping[str, str]], int]


def _run_interactive_shell(
    argv: tuple[str, ...],
    cwd: Path,
    environment: Mapping[str, str],
) -> int:
    """Run an interactive shell with inherited terminal streams and exact environment."""
    return subprocess.run(
        argv,
        cwd=cwd,
        env=dict(environment),
        check=False,
    ).returncode


def resolve_debug_shell(name: str) -> Path:
    """Resolve only the two explicitly supported interactive shell families."""
    if name not in {"zsh", "bash"}:
        raise ValueError("dev shell must be either zsh or bash")
    resolved = shutil.which(name, path="/bin:/usr/bin:/usr/local/bin:/opt/homebrew/bin")
    if resolved is None:
        raise ValueError(f"requested dev shell is unavailable: {name}")
    path = Path(resolved).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"requested dev shell is not executable: {path}")
    return path


class TemporaryOptimizerDevShell:
    """Open one disposable Optimizer workspace backed by temporary Gateway authority."""

    def __init__(
        self,
        artifacts: LocalArtifactStore,
        control: SqliteGatewayControl,
        sessions: CoreOptimizerSessionDriver,
        config: OptimizerSessionConfig,
        *,
        workspace_root: Path,
        gateway_endpoint: str,
        operations: frozenset[GatewayOperation],
        max_calls: int,
        capability_lifetime: timedelta,
        wiki_enabled: bool,
        shell_runner: ShellRunner = _run_interactive_shell,
    ) -> None:
        self._artifacts = artifacts
        self._control = control
        self._sessions = sessions
        self._config = config
        self._workspace_root = workspace_root.resolve()
        self._gateway_endpoint = gateway_endpoint
        self._operations = operations
        self._max_calls = max_calls
        self._capability_lifetime = capability_lifetime
        self._wiki_enabled = wiki_enabled
        self._shell_runner = shell_runner

    def open(
        self,
        *,
        shell_name: str,
        request: TemporaryOptimizerDevShellRequest,
    ) -> TemporaryDevShellResult:
        """Create, enter, revoke, and destroy one non-durable debug session."""
        shell = resolve_debug_shell(shell_name)
        capability = self._control.issue_bootstrap(
            BootstrapGatewaySubject(
                attempt_id=request.attempt_id,
                campaign_id=request.campaign_id,
                lineage_id=request.lineage_id,
                epoch_id=request.epoch_id,
                kernel_agent_revision_id=request.kernel_agent_revision_id,
                operator=request.operator,
                hardware_target=request.hardware_target,
                dsl=request.dsl,
                evaluation_contract_digest=request.evaluation_contract_digest,
                input_kernel_digest=request.input_kernel_digest,
                evidence_digest=request.epoch_evidence_checkpoint,
                created_at=datetime.now(UTC),
            ),
            GatewayCapabilityPolicy(
                operations=self._operations,
                max_calls=self._max_calls,
                expires_at=datetime.now(UTC) + self._capability_lifetime,
            ),
        )
        workspace: Path | None = None
        try:
            self._workspace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            workspace = Path(
                tempfile.mkdtemp(prefix="temporary-optimizer-", dir=self._workspace_root)
            )
            prepared = self._prepare(workspace, request)
            session_config = replace(
                self._config,
                model=request.model,
                gateway_endpoint=self._gateway_endpoint,
                gateway_capability=capability.token,
                wiki_endpoint=self._gateway_endpoint if self._wiki_enabled else None,
                wiki_capability=capability.token if self._wiki_enabled else None,
            )
            launch = self._sessions.prepare_launch(prepared, session_config)
            print(
                "\n".join(
                    (
                        "ATREX temporary Optimizer dev shell",
                        f"Attempt:   {request.attempt_id}",
                        f"DSL:       {request.dsl.value}",
                        f"Workspace: {workspace}",
                        "Campaign/Lineage: temporary; not registered",
                        "Agent backend: not started",
                        "Available CLI backends: claude, codex, qodercli, pi",
                        "Exit the shell to revoke access and destroy the workspace.",
                        "",
                    )
                ),
                flush=True,
            )
            shell_argv = self._sessions.wrap_command(launch, (str(shell), "-i"))
            returncode = self._shell_runner(shell_argv, prepared.root, {})
        finally:
            try:
                self._control.revoke(request.attempt_id)
            finally:
                if workspace is not None:
                    make_tree_owner_writable(workspace)
                    shutil.rmtree(workspace)
        assert workspace is not None
        return TemporaryDevShellResult(
            attempt_id=request.attempt_id,
            workspace=workspace,
            shell=shell,
            returncode=returncode,
        )

    def _prepare(
        self,
        root: Path,
        request: TemporaryOptimizerDevShellRequest,
    ) -> PreparedAttempt:
        manifest = AttemptInputManifestV9(
            attempt_id=request.attempt_id,
            kernel_agent_revision_id=request.kernel_agent_revision_id,
            input_kernel_revision_id=request.input_kernel_revision_id,
            input_kernel_digest=request.input_kernel_digest,
            epoch_evidence_checkpoint=request.epoch_evidence_checkpoint,
            attempt_evidence_digest=request.attempt_evidence_digest,
            optimizer_digest=request.optimizer_digest,
            dsl=request.dsl,
            context=AttemptTaskContextV5(
                campaign_id=request.campaign_id,
                lineage_id=request.lineage_id,
                epoch_id=request.epoch_id,
                epoch_number=1,
                attempt_ordinal=1,
                operator=request.operator,
                hardware_target=request.hardware_target,
                evaluation_contract_digest=request.evaluation_contract_digest,
                agent_problem_digest=request.agent_problem_digest,
            ),
        )
        paths = ATTEMPT_WORKSPACE_LAYOUT
        self._artifacts.materialize(request.input_kernel_digest, root / paths.input_kernel)
        checkpoint = self._artifacts.verify(request.epoch_evidence_checkpoint)
        if checkpoint.kind is not ArtifactKind.EVIDENCE:
            raise ValueError("temporary dev-shell checkpoint has the wrong Artifact kind")
        attempt_evidence = self._artifacts.verify(request.attempt_evidence_digest)
        if attempt_evidence.kind is not ArtifactKind.ATTEMPT_EVIDENCE:
            raise ValueError("temporary dev-shell Attempt Evidence has the wrong Artifact kind")
        assemble_optimizer_evidence_view(
            root / paths.evidence,
            control_root=root / ".runtime",
            lineage_payload=checkpoint.payload_path,
            lineage_checkpoint=request.epoch_evidence_checkpoint,
            attempt_payload=attempt_evidence.payload_path,
            attempt_snapshot=request.attempt_evidence_digest,
            current_epoch_number=1,
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            selected_revision=request.kernel_agent_revision_id,
            attempt_ordinal=1,
            artifacts=self._artifacts,
        )
        self._artifacts.materialize_file(
            request.agent_problem_digest,
            "value.json",
            root / paths.agent_problem,
        )
        self._artifacts.materialize(request.optimizer_digest, root / paths.optimizer)
        working_kernel = root / paths.working_kernel
        shutil.copytree(root / paths.input_kernel, working_kernel)
        make_tree_owner_writable(working_kernel)
        manifest_path = root / ATTEMPT_MANIFEST_RELATIVE_PATH
        manifest_path.parent.mkdir(mode=0o700, exist_ok=True)
        manifest_path.write_bytes(manifest.canonical_json_bytes())
        os.chmod(manifest_path, 0o400)
        session_root = root / "sessions"
        session_root.mkdir(mode=0o700)
        (root / "scratch").mkdir(mode=0o700)
        initialize_reusable_agent_state(root, root / paths.optimizer)
        remove_optimizer_state_seeds(root / paths.optimizer)
        return PreparedAttempt(
            root=root,
            manifest_path=manifest_path,
            session_root=session_root,
            session_id=f"temporary-dev-shell-{request.attempt_id}",
        )


class TemporaryEvolverDevShell:
    """Open one disposable Evolver workspace without durable lineage state."""

    def __init__(
        self,
        workspaces: EvolutionWorkspaceAssembler,
        sessions: SubprocessEvolutionSessionDriver,
        *,
        shell_runner: ShellRunner = _run_interactive_shell,
    ) -> None:
        self._workspaces = workspaces
        self._sessions = sessions
        self._shell_runner = shell_runner

    def open(
        self,
        *,
        shell_name: str,
        request: BuildChallengerRequest,
    ) -> TemporaryEvolverDevShellResult:
        """Create, enter, and destroy one synthetic Evolution snapshot."""
        shell = resolve_debug_shell(shell_name)
        prepared = self._workspaces.prepare(request)
        try:
            launch = self._sessions.prepare_launch(prepared)
            print(
                "\n".join(
                    (
                        "ATREX temporary Evolver dev shell",
                        f"Epoch:        {request.epoch_id} (synthetic)",
                        f"Parent Agent: {request.parent_revision.id} (agent-v0)",
                        f"DSL:          {request.parent_revision.dsl.value}",
                        f"Workspace:    {prepared.root}",
                        "Campaign/Lineage: temporary; not registered",
                        "Kernel history: empty; no measurements are fabricated",
                        "Agent backend: not started",
                        "Available CLI backends: claude, codex, qodercli, pi",
                        "Exit the shell to destroy the workspace.",
                        "",
                    )
                ),
                flush=True,
            )
            shell_argv = self._sessions.wrap_command(
                prepared,
                launch,
                (str(shell), "-i"),
            )
            returncode = self._shell_runner(shell_argv, prepared.root, {})
        finally:
            make_tree_owner_writable(prepared.root)
            shutil.rmtree(prepared.root)
        return TemporaryEvolverDevShellResult(
            epoch_id=request.epoch_id,
            parent_revision_id=request.parent_revision.id,
            workspace=prepared.root,
            shell=shell,
            returncode=returncode,
        )


class OptimizerDevShell:
    """Prepare one real Attempt run, inject authority, and launch no Agent backend."""

    def __init__(
        self,
        registry: SqliteRegistry,
        workspaces: LocalAttemptWorkspaceAssembler,
        evidence: LocalAttemptEvidenceAssembler,
        sessions: CoreOptimizerSessionDriver,
        authorities: AttemptTimedWorkerGatewayAuthorityProvider,
        leases: RegistryLineageLeaseManager,
        config: OptimizerSessionConfig,
        *,
        wiki_enabled: bool,
        shell_runner: ShellRunner = _run_interactive_shell,
        clock: Callable[[], str] = lambda: datetime.now(UTC).isoformat(),
    ) -> None:
        self._registry = registry
        self._workspaces = workspaces
        self._evidence = evidence
        self._sessions = sessions
        self._authorities = authorities
        self._leases = leases
        self._config = config
        self._wiki_enabled = wiki_enabled
        self._shell_runner = shell_runner
        self._clock = clock

    def open(
        self,
        *,
        shell_name: str,
        lineage_id: LineageId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> DevShellResult:
        """Create or reuse a running Attempt and retain its workspace after shell exit."""
        if (lineage_id is None) == (attempt_id is None):
            raise ValueError("dev shell requires exactly one lineage or Attempt target")
        if attempt_id is not None:
            initial = self._registry.get_attempt(attempt_id)
            target_lineage_id = self._registry.get_epoch(initial.epoch_id).lineage_id
        else:
            if lineage_id is None:
                raise AssertionError("validated dev-shell target is absent")
            target_lineage_id = lineage_id

        shell = resolve_debug_shell(shell_name)
        with self._leases.acquire(target_lineage_id):
            if attempt_id is None:
                attempt, created_attempt = self._ensure_first_active_attempt(target_lineage_id)
            else:
                attempt = self._registry.get_attempt(attempt_id)
                created_attempt = False
            epoch = self._registry.get_epoch(attempt.epoch_id)
            if epoch.lineage_id != target_lineage_id:
                raise ValueError("dev-shell Attempt moved to a different lineage")
            if attempt.status is not AttemptStatus.RUNNING:
                raise ValueError(f"dev-shell Attempt must be running, not {attempt.status.value}")
            if epoch.status not in {
                EpochStatus.BUILDING_CHALLENGER,
                EpochStatus.READY,
                EpochStatus.RUNNING,
            }:
                raise ValueError(f"dev-shell Epoch is not open for work: {epoch.status.value}")

            request = self._request(attempt, epoch)
            prepared = self._workspaces.prepare(request)
            authority = anyio.run(self._authorities.get_authority, request)
            session_config = replace(
                self._config,
                gateway_endpoint=authority.endpoint,
                gateway_capability=authority.capability,
                wiki_endpoint=authority.endpoint if self._wiki_enabled else None,
                wiki_capability=authority.capability if self._wiki_enabled else None,
            )
            launch = self._sessions.prepare_launch(prepared, session_config)
            event = {
                "worker_role": "optimizer-dev-shell",
                "worker_run_id": prepared.session_id,
                "lineage_id": target_lineage_id,
                "kernel_agent_revision_id": attempt.kernel_agent_revision_id,
                "input_kernel_revision_id": attempt.input_kernel_revision_id,
                "shell": str(shell),
            }
            self._registry.record_runtime_event("dev_shell.started", attempt.id, event)
            print(
                "\n".join(
                    (
                        "ATREX Optimizer dev shell",
                        f"Attempt:   {attempt.id}",
                        f"Lineage:   {target_lineage_id}",
                        f"Workspace: {prepared.root}",
                        "Agent backend: not started",
                        "Available CLI backends: claude, codex, qodercli, pi",
                        "Exit the shell to return; the workspace and running Attempt are retained.",
                        "",
                    )
                ),
                flush=True,
            )
            try:
                shell_argv = self._sessions.wrap_command(
                    launch,
                    (str(shell), "-i"),
                )
                returncode = self._shell_runner(
                    shell_argv,
                    prepared.root,
                    {},
                )
            except BaseException as error:
                self._registry.record_runtime_event(
                    "dev_shell.failed",
                    attempt.id,
                    {**event, "error_type": type(error).__name__},
                )
                raise
            finally:
                prepared.persist_reusable_directories()
            self._registry.record_runtime_event(
                "dev_shell.exited",
                attempt.id,
                {**event, "returncode": returncode},
            )
            return DevShellResult(
                attempt_id=attempt.id,
                lineage_id=target_lineage_id,
                workspace=prepared.root,
                shell=shell,
                returncode=returncode,
                created_attempt=created_attempt,
            )

    def _ensure_first_active_attempt(
        self,
        lineage_id: LineageId,
    ) -> tuple[Attempt, bool]:
        lineage = self._registry.get_lineage(lineage_id)
        epoch = self._registry.find_epoch(lineage.id, lineage.next_epoch_number)
        now = self._clock()
        if epoch is None:
            challenger_count = (
                0
                if lineage.next_epoch_number < lineage.challenger_start_epoch
                else lineage.challenger_count
            )
            epoch = Epoch(
                id=new_epoch_id(),
                lineage_id=lineage.id,
                number=lineage.next_epoch_number,
                active_kernel_agent_revision_id=lineage.active_kernel_agent_revision_id,
                challenger_kernel_agent_revision_ids=(),
                starting_kernel_revision_id=lineage.best_kernel_revision_id,
                evidence_checkpoint=lineage.evidence_checkpoint,
                challenger_count=challenger_count,
                trajectories_per_branch=lineage.trajectories_per_branch,
                attempts_per_trajectory=lineage.attempts_per_trajectory,
                status=(
                    EpochStatus.READY if challenger_count == 0 else EpochStatus.BUILDING_CHALLENGER
                ),
                winner_kernel_agent_revision_id=None,
                best_kernel_revision_id=None,
                created_at=now,
                completed_at=None,
            )
            self._registry.insert_epoch(epoch)
        attempt = self._registry.find_attempt(epoch.id, BranchRole.ACTIVE, 0, 1, 1)
        if attempt is not None:
            return attempt, False
        if epoch.status not in {
            EpochStatus.BUILDING_CHALLENGER,
            EpochStatus.READY,
            EpochStatus.RUNNING,
        }:
            raise ValueError(f"cannot create a dev-shell Attempt in Epoch {epoch.status.value}")
        attempt_id = new_attempt_id()
        evidence_request = BuildAttemptEvidenceRequest(
            attempt_id=attempt_id,
            epoch_id=epoch.id,
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=1,
            epoch_evidence_checkpoint=epoch.evidence_checkpoint,
        )
        attempt_evidence = self._evidence.assemble(evidence_request)
        attempt = Attempt(
            id=attempt_id,
            epoch_id=epoch.id,
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=1,
            kernel_agent_revision_id=epoch.active_kernel_agent_revision_id,
            input_kernel_revision_id=epoch.starting_kernel_revision_id,
            attempt_evidence_digest=attempt_evidence,
            output_kernel_revision_id=None,
            accepted_as_branch_best=False,
            status=AttemptStatus.RUNNING,
            infrastructure_failures=0,
            recovery_generation=0,
            authority_started_at=now,
            failure_reason=None,
            created_at=now,
            completed_at=None,
        )
        self._registry.insert_attempt(attempt)
        return attempt, True

    def _request(self, attempt: Attempt, epoch: Epoch) -> RunAttemptRequest:
        revision = self._registry.get_kernel_agent_revision(attempt.kernel_agent_revision_id)
        lineage = self._registry.get_lineage(epoch.lineage_id)
        return RunAttemptRequest(
            attempt_id=attempt.id,
            kernel_agent_revision_id=attempt.kernel_agent_revision_id,
            input_kernel_revision_id=attempt.input_kernel_revision_id,
            epoch_evidence_checkpoint=epoch.evidence_checkpoint,
            attempt_evidence_digest=attempt.attempt_evidence_digest,
            dsl=revision.dsl,
            model=lineage.optimizer_model,
        )


class EvolverDevShell:
    """Reconstruct one Epoch-scoped Evolver snapshot without starting its backend."""

    def __init__(
        self,
        registry: SqliteRegistry,
        workspaces: EvolutionWorkspaceAssembler,
        sessions: SubprocessEvolutionSessionDriver,
        leases: RegistryLineageLeaseManager,
        *,
        shell_runner: ShellRunner = _run_interactive_shell,
    ) -> None:
        self._registry = registry
        self._workspaces = workspaces
        self._sessions = sessions
        self._leases = leases
        self._shell_runner = shell_runner

    def open(
        self,
        *,
        shell_name: str,
        lineage_id: LineageId,
        epoch_number: int,
    ) -> EvolverDevShellResult:
        """Open a shell over the selected Epoch's frozen Evolution input boundary."""
        if epoch_number <= 0:
            raise ValueError("evolver dev shell Epoch number must be positive")
        shell = resolve_debug_shell(shell_name)
        with self._leases.acquire(lineage_id):
            self._registry.get_lineage(lineage_id)
            epoch = self._registry.find_epoch(lineage_id, epoch_number)
            if epoch is None:
                raise ValueError(
                    f"evolver dev shell Epoch {epoch_number} does not exist in {lineage_id}"
                )
            request = self._request(lineage_id, epoch)
            prepared = self._workspaces.prepare(request)
            launch = self._sessions.prepare_launch(prepared)
            event = {
                "worker_role": "evolver-dev-shell",
                "worker_run_id": prepared.root.name,
                "lineage_id": lineage_id,
                "epoch_id": epoch.id,
                "epoch_number": epoch.number,
                "epoch_status": epoch.status.value,
                "parent_revision_id": request.parent_revision.id,
                "shell": str(shell),
            }
            self._registry.record_runtime_event("dev_shell.started", epoch.id, event)
            print(
                "\n".join(
                    (
                        "ATREX Evolver dev shell",
                        f"Lineage:     {lineage_id}",
                        f"Epoch:       {epoch.number} ({epoch.status.value})",
                        f"Parent Agent: {request.parent_revision.id}",
                        f"Workspace:   {prepared.root}",
                        "Agent backend: not started",
                        "Available CLI backends: claude, codex, qodercli, pi",
                        "Snapshot excludes target-Epoch Attempt Kernels and all future Epochs.",
                        "Exit the shell to return; the frozen workspace is retained.",
                        "",
                    )
                ),
                flush=True,
            )
            try:
                shell_argv = self._sessions.wrap_command(
                    prepared,
                    launch,
                    (str(shell), "-i"),
                )
                returncode = self._shell_runner(
                    shell_argv,
                    prepared.root,
                    {},
                )
            except BaseException as error:
                self._registry.record_runtime_event(
                    "dev_shell.failed",
                    epoch.id,
                    {**event, "error_type": type(error).__name__},
                )
                raise
            self._registry.record_runtime_event(
                "dev_shell.exited",
                epoch.id,
                {**event, "returncode": returncode},
            )
            return EvolverDevShellResult(
                lineage_id=lineage_id,
                epoch_id=epoch.id,
                epoch_number=epoch.number,
                epoch_status=epoch.status,
                parent_revision_id=request.parent_revision.id,
                workspace=prepared.root,
                shell=shell,
                returncode=returncode,
            )

    def _request(self, lineage_id: LineageId, epoch: Epoch) -> BuildChallengerRequest:
        lineage = self._registry.get_lineage(lineage_id)
        parent = self._registry.get_kernel_agent_revision(epoch.active_kernel_agent_revision_id)
        current_challengers = set(epoch.challenger_kernel_agent_revision_ids)
        agent_catalog = tuple(
            replace(
                entry,
                disposition=(
                    "challenger" if entry.revision.id in current_challengers else entry.disposition
                ),
                active=entry.revision.id == parent.id,
            )
            for entry in self._registry.list_lineage_agent_revisions(lineage_id)
            if entry.introduced_epoch_number is None
            or entry.introduced_epoch_number < epoch.number
            or entry.revision.id in current_challengers
        )
        if not any(entry.revision.id == parent.id for entry in agent_catalog):
            raise RuntimeError("selected Epoch parent Agent is absent from its lineage catalog")
        kernel_catalog = tuple(
            entry
            for entry in self._registry.list_lineage_kernels(lineage_id)
            if entry.epoch_number is None or entry.epoch_number < epoch.number
        )
        return BuildChallengerRequest(
            parent_revision=parent,
            epoch_id=epoch.id,
            evidence_checkpoint=epoch.evidence_checkpoint,
            idempotency_key=f"evolver-dev-shell:{epoch.id}",
            agent_catalog=agent_catalog,
            kernel_catalog=kernel_catalog,
            model=lineage.evolver_model,
        )


def build_dev_shell_evidence(
    registry: SqliteRegistry,
    artifacts: LocalArtifactStore,
    limits: EvidenceProjectionLimits,
    *,
    redaction_patterns: tuple[str, ...],
) -> LocalAttemptEvidenceAssembler:
    """Compose the same branch-local Evidence projection used by Campaign workers."""
    return LocalAttemptEvidenceAssembler(
        registry,
        artifacts,
        EvidenceArtifactProjector(
            artifacts,
            limits,
            redaction_patterns=redaction_patterns,
        ),
    )


__all__ = [
    "DevShellResult",
    "EvolverDevShell",
    "EvolverDevShellResult",
    "OptimizerDevShell",
    "ShellRunner",
    "TemporaryEvolverDevShell",
    "TemporaryEvolverDevShellResult",
    "TemporaryOptimizerDevShell",
    "TemporaryOptimizerDevShellRequest",
    "build_dev_shell_evidence",
    "create_empty_attempt_evidence",
    "resolve_debug_shell",
]
