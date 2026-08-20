"""Interactive Optimizer dev-shell lifecycle tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import NOW, digest, seed_lineage

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.controller.leases import RegistryLineageLeaseManager
from atrex_runtime.dev_shell import (
    EvolverDevShell,
    OptimizerDevShell,
    TemporaryEvolverDevShell,
    TemporaryOptimizerDevShell,
    TemporaryOptimizerDevShellRequest,
    create_empty_attempt_evidence,
)
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
)
from atrex_runtime.domain.models import AttemptStatus, Dsl, Epoch, EpochStatus, LineageStatus
from atrex_runtime.gateway.control_models import GatewayCapability, GatewayOperation
from atrex_runtime.ports import (
    BuildAttemptEvidenceRequest,
    BuildChallengerRequest,
    RunAttemptRequest,
    WorkerGatewayAuthority,
)
from atrex_runtime.registry.sqlite import SqliteRegistry
from atrex_runtime.workers.evolution import PreparedEvolution
from atrex_runtime.workers.optimizer import OptimizerSessionConfig
from atrex_runtime.workers.workspace import PreparedAttempt


class FakeEvidence:
    """Return a valid identity without materializing projection input in this unit test."""

    def assemble(self, request: BuildAttemptEvidenceRequest) -> ArtifactDigest:
        return digest(f"dev-evidence:{request.attempt_id}")


class FakeWorkspaces:
    """Allocate a distinct retained run directory and capture the trusted request."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests: list[RunAttemptRequest] = []

    def prepare(self, request: RunAttemptRequest) -> PreparedAttempt:
        self.requests.append(request)
        run = self.root / f"run-{len(self.requests)}"
        run.mkdir(parents=True)
        manifest = run / "attempt.json"
        manifest.write_text("{}", encoding="utf-8")
        sessions = run / "sessions"
        sessions.mkdir()
        return PreparedAttempt(run, manifest, sessions, f"dev-{len(self.requests)}")


class FakeSessions:
    """Expose the final launch environment without executing the Core entrypoint."""

    def prepare_launch(
        self,
        prepared: PreparedAttempt,
        config: OptimizerSessionConfig,
    ) -> SimpleNamespace:
        assert config.gateway_endpoint == "http://runtime.test"
        assert config.gateway_capability == "attempt-capability"
        assert config.wiki_endpoint == "http://runtime.test"
        assert config.wiki_capability == "attempt-capability"
        return SimpleNamespace(
            environment={
                **dict(config.environment),
                "ATREX_GATEWAY_PROXY_URL": config.gateway_endpoint,
                "ATREX_GATEWAY_CAPABILITY": config.gateway_capability,
                "ATREX_WIKI_PROXY_URL": config.wiki_endpoint,
                "ATREX_WIKI_CAPABILITY": config.wiki_capability,
                "ATREX_OPTIMIZER_REPOSITORY": str(prepared.root / "agent/optimizer"),
            }
        )

    def wrap_command(
        self,
        launch: SimpleNamespace,
        runtime_argv: tuple[str, ...],
    ) -> tuple[str, ...]:
        assert launch.environment["ATREX_GATEWAY_CAPABILITY"] == "attempt-capability"
        return runtime_argv


class FakeAuthorities:
    """Return one Attempt-scoped authority exactly as the production provider does."""

    async def get_authority(self, request: RunAttemptRequest) -> WorkerGatewayAuthority:
        assert request.attempt_id
        return WorkerGatewayAuthority("http://runtime.test", "attempt-capability")


class FakeEvolutionWorkspaces:
    """Capture the frozen Epoch request while allocating a retained workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests: list[BuildChallengerRequest] = []

    def prepare(self, request: BuildChallengerRequest) -> PreparedEvolution:
        self.requests.append(request)
        run = self.root / f"run-{len(self.requests)}"
        (run / "candidate").mkdir(parents=True)
        (run / "scratch").mkdir()
        manifest = run / "evolution-input.json"
        manifest.write_text("{}", encoding="utf-8")
        return PreparedEvolution(
            root=run,
            manifest_path=manifest,
            candidate_root=run / "candidate",
            output_path=run / "scratch/evolution-output.json",
            parent_revision=request.parent_revision,
        )


class FakeEvolutionSessions:
    """Expose the Evolver shell environment without starting its backend."""

    def prepare_launch(self, prepared: PreparedEvolution) -> SimpleNamespace:
        return SimpleNamespace(
            environment={
                "PATH": "/usr/bin:/bin",
                "ATREX_EVOLUTION_INPUT": str(prepared.manifest_path),
                "ATREX_EVOLUTION_CANDIDATE": str(prepared.candidate_root),
            }
        )

    def wrap_command(
        self,
        prepared: PreparedEvolution,
        launch: SimpleNamespace,
        runtime_argv: tuple[str, ...],
    ) -> tuple[str, ...]:
        assert launch.environment["ATREX_EVOLUTION_INPUT"] == str(prepared.manifest_path)
        return runtime_argv


class FakeTemporaryControl:
    def __init__(self) -> None:
        self.revoked = []

    def issue_bootstrap(self, subject: object, policy: object) -> GatewayCapability:
        del policy
        return GatewayCapability("temporary-capability", subject.attempt_id)  # type: ignore[attr-defined]

    def revoke(self, attempt_id: object) -> None:
        self.revoked.append(attempt_id)


class FakeTemporarySessions:
    def prepare_launch(
        self,
        prepared: PreparedAttempt,
        config: OptimizerSessionConfig,
    ) -> SimpleNamespace:
        assert config.gateway_capability == "temporary-capability"
        assert (prepared.root / "attempt.json").is_file()
        assert (prepared.root / "work/kernel/kernel.py").is_file()
        return SimpleNamespace(environment={})

    def wrap_command(
        self,
        launch: SimpleNamespace,
        runtime_argv: tuple[str, ...],
    ) -> tuple[str, ...]:
        del launch
        return runtime_argv


def test_dev_shell_creates_real_attempt_and_retains_workspace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "registry.sqlite"
    with SqliteRegistry(database) as bootstrap_registry:
        seeded = seed_lineage(bootstrap_registry, attempts_per_trajectory=2)
    registry = SqliteRegistry(database, require_fencing=True)
    try:
        workspaces = FakeWorkspaces(tmp_path / "workspaces")
        launches: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

        def run_shell(
            argv: tuple[str, ...],
            cwd: Path,
            environment: Mapping[str, str],
        ) -> int:
            values = dict(environment)
            launches.append((argv, cwd, values))
            assert values == {}
            return 7

        service = OptimizerDevShell(
            registry,
            workspaces,  # type: ignore[arg-type]
            FakeEvidence(),  # type: ignore[arg-type]
            FakeSessions(),  # type: ignore[arg-type]
            FakeAuthorities(),  # type: ignore[arg-type]
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=30,
                heartbeat_seconds=5,
            ),
            OptimizerSessionConfig(environment=(("PATH", "/usr/bin:/bin"),)),
            wiki_enabled=True,
            shell_runner=run_shell,
            clock=lambda: NOW,
        )

        result = service.open(shell_name="bash", lineage_id=seeded.lineage_id)

        assert result.created_attempt
        assert result.returncode == 7
        assert result.workspace.is_dir()
        assert launches[0][0] == (str(result.shell), "-i")
        assert launches[0][1] == result.workspace
        assert workspaces.requests[0].attempt_id == result.attempt_id
        assert registry.get_attempt(result.attempt_id).status is AttemptStatus.RUNNING
        assert registry.get_lineage(seeded.lineage_id).status is LineageStatus.RUNNING

        resumed = service.open(shell_name="bash", attempt_id=result.attempt_id)

        assert not resumed.created_attempt
        assert resumed.attempt_id == result.attempt_id
        assert resumed.workspace != result.workspace
        assert resumed.workspace.is_dir()
        assert registry.get_attempt(result.attempt_id).status is AttemptStatus.RUNNING
        assert len(workspaces.requests) == 2
        assert len(launches) == 2
        output = capsys.readouterr().out
        assert "ATREX Optimizer dev shell" in output
        assert "attempt-capability" not in output
    finally:
        registry.close()


def test_temporary_dev_shell_skips_registry_and_destroys_workspace(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    lineage_id = new_lineage_id()
    epoch_id = new_epoch_id()
    attempt_id = new_attempt_id()

    kernel = tmp_path / "kernel"
    kernel.mkdir()
    (kernel / "kernel.py").write_text("# temporary kernel\n", encoding="utf-8")
    input_kernel = artifacts.put_directory(kernel, ArtifactKind.KERNEL)
    optimizer = tmp_path / "optimizer"
    optimizer.mkdir()
    (optimizer / "atrex-bundle.json").write_text("{}", encoding="utf-8")
    optimizer_digest = artifacts.put_directory(optimizer, ArtifactKind.KERNEL_AGENT)
    problem_digest = artifacts.put_json({}, ArtifactKind.AGENT_PROBLEM)
    contract_digest = artifacts.put_json({}, ArtifactKind.EVALUATION_CONTRACT)

    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / "bootstrap").mkdir(parents=True)
    (checkpoint_root / "bootstrap-metadata.json").write_text(
        '{"schema_version":1,"source":"test"}', encoding="utf-8"
    )
    (checkpoint_root / "checkpoint.json").write_text(
        '{"schema_version":1,"lineage_id":"'
        + str(lineage_id)
        + '","through_epoch":0,"previous_checkpoint_digest":null}',
        encoding="utf-8",
    )
    checkpoint = artifacts.put_directory(checkpoint_root, ArtifactKind.EVIDENCE)
    attempt_evidence = create_empty_attempt_evidence(
        artifacts,
        epoch_id=epoch_id,
        attempt_id=attempt_id,
        epoch_evidence_checkpoint=checkpoint,
    )
    control = FakeTemporaryControl()
    seen_workspace: list[Path] = []

    def run_shell(
        argv: tuple[str, ...],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> int:
        assert argv[-1] == "-i"
        assert environment == {}
        assert cwd.is_dir()
        seen_workspace.append(cwd)
        return 0

    service = TemporaryOptimizerDevShell(
        artifacts,
        control,  # type: ignore[arg-type]
        FakeTemporarySessions(),  # type: ignore[arg-type]
        OptimizerSessionConfig(environment=()),
        workspace_root=tmp_path / "workspaces",
        gateway_endpoint="http://runtime.test",
        operations=frozenset({GatewayOperation.HEALTH}),
        max_calls=1,
        capability_lifetime=timedelta(minutes=5),
        wiki_enabled=False,
        shell_runner=run_shell,
    )
    result = service.open(
        shell_name="bash",
        request=TemporaryOptimizerDevShellRequest(
            attempt_id=attempt_id,
            campaign_id=new_campaign_id(),
            lineage_id=lineage_id,
            epoch_id=epoch_id,
            kernel_agent_revision_id=new_kernel_agent_revision_id(),
            input_kernel_revision_id=new_kernel_revision_id(),
            optimizer_digest=optimizer_digest,
            input_kernel_digest=input_kernel,
            evaluation_contract_digest=contract_digest,
            agent_problem_digest=problem_digest,
            epoch_evidence_checkpoint=checkpoint,
            attempt_evidence_digest=attempt_evidence,
            dsl=Dsl.TRITON,
            operator="vector_add",
            hardware_target="H100",
        ),
    )

    assert result.returncode == 0
    assert seen_workspace == [result.workspace]
    assert not result.workspace.exists()
    assert control.revoked == [attempt_id]


def test_temporary_evolver_dev_shell_destroys_synthetic_workspace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "registry.sqlite"
    with SqliteRegistry(database) as registry:
        seeded = seed_lineage(registry)
        parent = registry.get_kernel_agent_revision(seeded.active_revision_id)
    workspaces = FakeEvolutionWorkspaces(tmp_path / "evolutions")
    seen_workspace: list[Path] = []

    def run_shell(
        argv: tuple[str, ...],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> int:
        assert argv[-1] == "-i"
        assert environment == {}
        assert cwd.is_dir()
        seen_workspace.append(cwd)
        return 3

    epoch_id = new_epoch_id()
    service = TemporaryEvolverDevShell(
        workspaces,  # type: ignore[arg-type]
        FakeEvolutionSessions(),  # type: ignore[arg-type]
        shell_runner=run_shell,
    )
    result = service.open(
        shell_name="bash",
        request=BuildChallengerRequest(
            parent_revision=parent,
            epoch_id=epoch_id,
            evidence_checkpoint=digest("temporary-evolver-evidence"),
            idempotency_key=f"temporary-evolver:{epoch_id}",
        ),
    )

    assert result.returncode == 3
    assert result.epoch_id == epoch_id
    assert result.parent_revision_id == parent.id
    assert seen_workspace == [result.workspace]
    assert not result.workspace.exists()
    output = capsys.readouterr().out
    assert "ATREX temporary Evolver dev shell" in output
    assert "not registered" in output
    assert "Kernel history: empty" in output


def test_evolver_dev_shell_reconstructs_selected_epoch_without_mutating_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "registry.sqlite"
    with SqliteRegistry(database) as bootstrap_registry:
        seeded = seed_lineage(bootstrap_registry, challenger_count=0)
        lineage = bootstrap_registry.get_lineage(seeded.lineage_id)
        epoch = Epoch(
            id=new_epoch_id(),
            lineage_id=lineage.id,
            number=1,
            active_kernel_agent_revision_id=lineage.active_kernel_agent_revision_id,
            challenger_kernel_agent_revision_ids=(),
            starting_kernel_revision_id=lineage.best_kernel_revision_id,
            evidence_checkpoint=lineage.evidence_checkpoint,
            challenger_count=0,
            trajectories_per_branch=lineage.trajectories_per_branch,
            attempts_per_trajectory=lineage.attempts_per_trajectory,
            status=EpochStatus.READY,
            winner_kernel_agent_revision_id=None,
            best_kernel_revision_id=None,
            created_at=NOW,
            completed_at=None,
        )
        bootstrap_registry.insert_epoch(epoch)

    registry = SqliteRegistry(database, require_fencing=True)
    try:
        workspaces = FakeEvolutionWorkspaces(tmp_path / "evolutions")
        launches: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

        def run_shell(
            argv: tuple[str, ...],
            cwd: Path,
            environment: Mapping[str, str],
        ) -> int:
            launches.append((argv, cwd, dict(environment)))
            return 9

        service = EvolverDevShell(
            registry,
            workspaces,  # type: ignore[arg-type]
            FakeEvolutionSessions(),  # type: ignore[arg-type]
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=30,
                heartbeat_seconds=5,
            ),
            shell_runner=run_shell,
        )

        result = service.open(
            shell_name="bash",
            lineage_id=seeded.lineage_id,
            epoch_number=1,
        )

        request = workspaces.requests[0]
        assert result.epoch_id == epoch.id
        assert result.epoch_number == 1
        assert result.epoch_status is EpochStatus.READY
        assert result.parent_revision_id == seeded.active_revision_id
        assert result.returncode == 9
        assert request.epoch_id == epoch.id
        assert request.evidence_checkpoint == epoch.evidence_checkpoint
        assert [entry.revision.id for entry in request.agent_catalog] == [seeded.active_revision_id]
        assert request.agent_catalog[0].active
        assert [entry.revision.id for entry in request.kernel_catalog] == [seeded.baseline.id]
        assert registry.list_attempts(epoch.id) == []
        assert launches[0][1] == result.workspace
        assert launches[0][2] == {}
        output = capsys.readouterr().out
        assert "ATREX Evolver dev shell" in output
        assert "Agent backend: not started" in output
        assert "Available CLI backends: claude, codex, qodercli, pi" in output
    finally:
        registry.close()


def test_evolver_dev_shell_requires_existing_epoch(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite"
    with SqliteRegistry(database) as bootstrap_registry:
        seeded = seed_lineage(bootstrap_registry)
    registry = SqliteRegistry(database, require_fencing=True)
    try:
        service = EvolverDevShell(
            registry,
            FakeEvolutionWorkspaces(tmp_path / "evolutions"),  # type: ignore[arg-type]
            FakeEvolutionSessions(),  # type: ignore[arg-type]
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=30,
                heartbeat_seconds=5,
            ),
        )
        with pytest.raises(ValueError, match="does not exist"):
            service.open(
                shell_name="bash",
                lineage_id=seeded.lineage_id,
                epoch_number=1,
            )
    finally:
        registry.close()
