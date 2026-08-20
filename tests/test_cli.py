"""Runtime administration CLI tests."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from conftest import NOW, digest, seed_lineage

import atrex_runtime.cli as runtime_cli
import atrex_runtime.cli.campaign as runtime_cli_campaign
import atrex_runtime.cli.maintenance as runtime_cli_maintenance
import atrex_runtime.cli.progress as runtime_cli_progress
from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.controller import CampaignScheduleResult, LineageScheduleResult
from atrex_runtime.domain.ids import (
    LineageId,
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
    new_wiki_feedback_id,
    new_worker_session_id,
)
from atrex_runtime.domain.models import (
    Attempt,
    AttemptReportStatus,
    AttemptStatus,
    BranchRole,
    ChallengerProposalType,
    Dsl,
    Epoch,
    EpochChallenger,
    EpochSelection,
    EpochStatus,
    KernelAgentRevision,
    KernelMeasurement,
    KernelMeasurementPurpose,
    Lineage,
    LineageStatus,
    TokenUsage,
    WikiFeedbackStatus,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)
from atrex_runtime.gateway.control import (
    BootstrapGatewaySubject,
    BootstrapRunStatus,
    GatewayCapabilityPolicy,
    GatewayEvaluationSource,
    GatewayOperation,
    SqliteGatewayControl,
)
from atrex_runtime.knowledge import WikiFeedbackDrainResult
from atrex_runtime.registry.sqlite import SqliteRegistry


@dataclass
class RecordingScheduler:
    """Return one fixed final state while recording the absolute target."""

    lineage: Lineage
    target: int | None = None

    async def run_campaign_through(
        self,
        lineage_ids: tuple[LineageId, ...],
        target_epoch_number: int,
    ) -> CampaignScheduleResult:
        assert lineage_ids == (self.lineage.id,)
        self.target = target_epoch_number
        return CampaignScheduleResult(
            self.lineage.campaign_id,
            (LineageScheduleResult(self.lineage, (2, 3)),),
        )


class FakeCampaignRuntime:
    """Minimal context-managed scheduler owner accepted by the CLI."""

    def __init__(self, scheduler: RecordingScheduler) -> None:
        self.scheduler = scheduler
        self.closed = False

    def __enter__(self) -> FakeCampaignRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.closed = True


class RecordingDrainer:
    """Return one bounded feedback result and record invocation."""

    def __init__(self) -> None:
        self.called = False

    async def drain_once(self) -> WikiFeedbackDrainResult:
        self.called = True
        return WikiFeedbackDrainResult(3, 2, 1, 0)


class FakeWikiFeedbackRuntime:
    """Minimal context-managed Drainer owner accepted by the CLI."""

    def __init__(self, drainer: RecordingDrainer) -> None:
        self.drainer = drainer
        self.poll_seconds = 1.0
        self.closed = False
        self.requeued = None
        self.compact = None

    def __enter__(self) -> FakeWikiFeedbackRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.closed = True

    def requeue(self, item_id: object) -> object:
        self.requeued = item_id
        return type("Item", (), {"id": item_id, "status": WikiFeedbackStatus.PENDING})()

    def maintain(self, *, compact: bool) -> int:
        self.compact = compact
        return 7


def _server_only_config(tmp_path: Path) -> Path:
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server": {"host": "127.0.0.1", "port": 8765},
                "storage": {
                    "registry_database": "state/registry.sqlite",
                    "gateway_database": "state/gateway.sqlite",
                    "agate_jobs_database": "state/agate-jobs.sqlite",
                    "artifacts_root": "state/artifacts",
                },
                "gateway_proxy": {
                    "max_request_bytes": 65536,
                    "max_candidate_files": 16,
                    "max_candidate_bytes": 32768,
                    "capability_signing_key_env": "UNUSED_SIGNING_KEY",
                    "candidate_diff_allowed_paths": {
                        "cuda": ["*.cu"],
                        "triton": ["*.py"],
                        "cutedsl": ["*.py"],
                    },
                    "candidate_diff_require_change": True,
                },
                "agate": {
                    "base_url": "https://gateway.example.test",
                    "auth_mode": "none",
                    "http_timeout_s": 60,
                    "wait_timeout_s": 900,
                },
                "kernel_agent": {
                    "max_bundle_files": 128,
                    "max_bundle_bytes": 65536,
                    "max_entrypoint_bytes": 8192,
                    "max_agent_problem_bytes": 4096,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_run_campaign_cli_reports_recoverable_target_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = Lineage(
        id=new_lineage_id(),
        campaign_id=new_campaign_id(),
        dsl=Dsl.TRITON,
        hardware_target="nvidia-h100",
        active_kernel_agent_revision_id=new_kernel_agent_revision_id(),
        best_kernel_revision_id=new_kernel_revision_id(),
        evidence_checkpoint="sha256:" + "a" * 64,
        challenger_count=1,
        trajectories_per_branch=1,
        attempts_per_trajectory=4,
        next_epoch_number=4,
        status=LineageStatus.READY,
    )
    scheduler = RecordingScheduler(lineage)
    runtime = FakeCampaignRuntime(scheduler)

    def build_fake(
        settings: RuntimeSettings,
        environment: object,
        *,
        attempt_finished: object = None,
    ) -> FakeCampaignRuntime:
        del settings, environment
        assert callable(attempt_finished)
        return runtime

    monkeypatch.setattr(runtime_cli_campaign, "build_campaign_runtime", build_fake)

    runtime_cli.main(
        [
            "run-campaign",
            "--config",
            str(_server_only_config(tmp_path)),
            "--lineage",
            str(lineage.id),
            "--target-epoch",
            "3",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert scheduler.target == 3
    assert runtime.closed
    assert output["campaign_id"] == lineage.campaign_id
    assert output["lineages"][0]["completed_epochs"] == [2, 3]
    assert output["lineages"][0]["next_epoch_number"] == 4


def test_attempt_progress_distinguishes_agent_branch_and_trajectory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    epoch = cast(
        Epoch,
        SimpleNamespace(
            id=new_epoch_id(),
            lineage_id=new_lineage_id(),
            number=1,
            challenger_count=1,
            trajectories_per_branch=3,
            attempts_per_trajectory=3,
        ),
    )
    active = cast(
        Attempt,
        SimpleNamespace(
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=1,
            completed_at="2026-08-18T12:01:02+00:00",
        ),
    )
    challenger = cast(
        Attempt,
        SimpleNamespace(
            branch=BranchRole.CHALLENGER,
            challenger_ordinal=1,
            trajectory_ordinal=1,
            ordinal=2,
            completed_at="2026-08-18T12:03:04+00:00",
        ),
    )

    runtime_cli_progress.print_attempt_finished(epoch, active)
    runtime_cli_progress.print_attempt_finished(epoch, challenger)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "[2026-08-18T12:01:02+00:00] active trajectory 1 attempt 1 finished",
        "[2026-08-18T12:03:04+00:00] challenger-1 trajectory 1 attempt 2 finished",
    ]


def test_interactive_attempt_progress_draws_every_branch() -> None:
    stream = io.StringIO()
    renderer = runtime_cli_progress.AttemptProgressRenderer(stream, interactive=True)
    epoch = cast(
        Epoch,
        SimpleNamespace(
            id=new_epoch_id(),
            lineage_id=new_lineage_id(),
            number=1,
            challenger_count=1,
            trajectories_per_branch=2,
            attempts_per_trajectory=3,
        ),
    )
    attempt = cast(
        Attempt,
        SimpleNamespace(
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=1,
            completed_at="2026-08-18T12:01:02+00:00",
        ),
    )

    renderer(epoch, attempt)

    output = stream.getvalue()
    assert "Epoch 1 branch progress" in output
    assert "  active\n" in output
    assert "  challenger-1\n" in output
    assert output.count("    trajectory 1") == 2
    assert output.count("    trajectory 2") == 2
    assert "[█░░] 1/3" in output
    assert output.count("[░░░] 0/3") == 3


def test_run_campaign_cli_rejects_invalid_lineage_before_composition(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid lineage identifier"):
        runtime_cli.main(
            [
                "run-campaign",
                "--config",
                str(_server_only_config(tmp_path)),
                "--lineage",
                "not-a-lineage",
                "--target-epoch",
                "1",
            ]
        )


def test_digest_evolver_bundle_cli_reports_validated_content_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "evolver"
    bundle.mkdir()
    (bundle / "atrex-evolver-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-evolver-bundle-v1",
                "entrypoint": {"command": "main.py"},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "main.py").write_text("print('evolve')\n", encoding="utf-8")

    runtime_cli.main(["digest-evolver-bundle", "--path", str(bundle)])

    output = json.loads(capsys.readouterr().out)
    assert len(output["bundle_sha256"]) == 64


def test_dev_shell_cli_dispatches_lineage_without_starting_campaign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def open_shell(
        config_path: str,
        lineage_value: str | None,
        attempt_value: str | None,
        shell_name: str,
    ) -> None:
        captured.update(
            config=config_path,
            lineage=lineage_value,
            attempt=attempt_value,
            shell=shell_name,
        )

    monkeypatch.setattr(runtime_cli, "open_optimizer_dev_shell", open_shell)
    runtime_cli.main(
        [
            "dev-shell",
            "--config",
            "runtime.json",
            "--lineage",
            "lineage_" + "1" * 32,
            "--shell",
            "bash",
        ]
    )

    assert captured == {
        "config": "runtime.json",
        "lineage": "lineage_" + "1" * 32,
        "attempt": None,
        "shell": "bash",
    }


def test_evolver_dev_shell_cli_requires_lineage_and_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def open_shell(
        config_path: str,
        lineage_value: str,
        epoch_number: int,
        shell_name: str,
    ) -> None:
        captured.update(
            config=config_path,
            lineage=lineage_value,
            epoch=epoch_number,
            shell=shell_name,
        )

    monkeypatch.setattr(runtime_cli, "open_evolver_dev_shell", open_shell)
    runtime_cli.main(
        [
            "evolver-dev-shell",
            "--config",
            "runtime.json",
            "--lineage",
            "lineage_" + "2" * 32,
            "--epoch",
            "3",
            "--shell",
            "bash",
        ]
    )

    assert captured == {
        "config": "runtime.json",
        "lineage": "lineage_" + "2" * 32,
        "epoch": 3,
        "shell": "bash",
    }


def test_temporary_evolver_dev_shell_cli_requires_campaign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def open_shell(config_path: str, campaign_path: str, shell_name: str) -> None:
        captured.update(config=config_path, campaign=campaign_path, shell=shell_name)

    monkeypatch.setattr(runtime_cli, "open_temporary_evolver_dev_shell", open_shell)
    runtime_cli.main(
        [
            "temporary-evolver-dev-shell",
            "--config",
            "runtime.json",
            "--campaign",
            "campaign.json",
            "--shell",
            "bash",
        ]
    )

    assert captured == {
        "config": "runtime.json",
        "campaign": "campaign.json",
        "shell": "bash",
    }


def test_kernel_catalog_cli_lists_and_shows_agent_measurements(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _server_only_config(tmp_path)
    settings = RuntimeSettings.from_file(config)
    artifacts = LocalArtifactStore(settings.storage.artifacts_root)
    gateway_result = artifacts.put_json(
        {"result": {"performance": {"shapes": {"0": {"sol": {"pct": 62.5}}}}}},
        ArtifactKind.GATEWAY_RESULT,
    )
    with SqliteRegistry(settings.storage.registry_database) as registry:
        seeded = seed_lineage(registry, gateway_result_digest=gateway_result)
        lineage = registry.get_lineage(seeded.lineage_id)
        registry.record_kernel_measurement(
            KernelMeasurement(
                id="measurement-cli",
                kernel_revision_id=seeded.baseline.id,
                purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
                repeat=0,
                correct=True,
                latency_us=98.0,
                gateway_result_digest=digest("measurement-cli-result"),
                agate_job_id="agate-cli",
                created_at=NOW,
            )
        )

    runtime_cli.main(
        [
            "list-kernels",
            "--config",
            str(config),
            "--campaign",
            str(lineage.campaign_id),
        ]
    )
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["kernels"][0]["kernel_revision_id"] == seeded.baseline.id
    assert catalog["kernels"][0]["kernel_agent_revision_id"] == seeded.active_revision_id
    assert catalog["kernels"][0]["version"] == "v0"
    assert catalog["kernels"][0]["parent_version"] is None
    assert catalog["kernels"][0]["disposition"] == "baseline"
    assert catalog["kernels"][0]["sol_percent"] == 62.5
    assert catalog["kernels"][0]["kernel_artifact"]["referenced_at"] == NOW

    runtime_cli.main(
        [
            "list-kernels",
            "--config",
            str(config),
            "--lineage",
            str(seeded.lineage_id),
            "--format",
            "table",
        ]
    )
    table = capsys.readouterr().out
    assert "VERSION" in table
    assert "PARENT" in table
    assert "SOL_%" in table
    assert "62.500" in table
    assert "v0" in table
    assert "baseline" in table

    runtime_cli.main(
        [
            "show-kernel",
            "--config",
            str(config),
            "--kernel",
            str(seeded.baseline.id),
        ]
    )
    detail = json.loads(capsys.readouterr().out)
    assert detail["version"] == "v0"
    assert detail["kernel_agent_version"] == "agent-v0"
    assert detail["sol_percent"] == 62.5
    assert [item["purpose"] for item in detail["measurements"]] == [
        "framework_baseline",
        "kernel_retention",
    ]

    runtime_cli.main(
        [
            "list-agent-revisions",
            "--config",
            str(config),
            "--lineage",
            str(seeded.lineage_id),
        ]
    )
    agents = json.loads(capsys.readouterr().out)
    assert agents["agent_revisions"][0]["agent_version"] == "agent-v0"
    assert agents["agent_revisions"][0]["active"] is True

    runtime_cli.main(
        [
            "show-agent-revision",
            "--config",
            str(config),
            "--agent-revision",
            str(seeded.active_revision_id),
        ]
    )
    agent = json.loads(capsys.readouterr().out)
    assert agent["parent_agent_version"] is None
    assert agent["optimizer_artifact"]["referenced_at"] == NOW

    runtime_cli.main(
        [
            "list-agent-revisions",
            "--config",
            str(config),
            "--lineage",
            str(seeded.lineage_id),
            "--format",
            "table",
        ]
    )
    agent_table = capsys.readouterr().out
    assert "AGENT_VERSION" in agent_table
    assert "agent-v0" in agent_table


def test_epoch_history_cli_shows_challenger_promotion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _server_only_config(tmp_path)
    settings = RuntimeSettings.from_file(config)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        seeded = seed_lineage(registry, attempts_per_trajectory=1)
        lineage = registry.get_lineage(seeded.lineage_id)
        challenger = registry.register_kernel_agent_revision(
            KernelAgentRevision(
                id=new_kernel_agent_revision_id(),
                parent_id=seeded.active_revision_id,
                creation_key="cli-epoch-challenger",
                dsl=Dsl.TRITON,
                optimizer_digest=digest("cli-epoch-challenger"),
                created_by="evolver",
                created_at=NOW,
                evolution_trace_digest=digest("cli-epoch-evolution"),
            )
        )
        epoch = Epoch(
            id=new_epoch_id(),
            lineage_id=lineage.id,
            number=1,
            active_kernel_agent_revision_id=seeded.active_revision_id,
            challenger_kernel_agent_revision_ids=(),
            starting_kernel_revision_id=seeded.baseline.id,
            evidence_checkpoint=lineage.evidence_checkpoint,
            challenger_count=1,
            trajectories_per_branch=1,
            attempts_per_trajectory=1,
            status=EpochStatus.BUILDING_CHALLENGER,
            winner_kernel_agent_revision_id=None,
            best_kernel_revision_id=None,
            created_at=NOW,
            completed_at=None,
        )
        registry.insert_epoch(epoch)
        registry.attach_challenger(
            EpochChallenger(
                epoch.id,
                1,
                challenger.id,
                ChallengerProposalType.EVOLVED,
                seeded.active_revision_id,
                digest("cli-epoch-evolution"),
            )
        )
        registry.transition_epoch(epoch.id, EpochStatus.READY, EpochStatus.RUNNING)
        registry.transition_epoch(epoch.id, EpochStatus.RUNNING, EpochStatus.SELECTING)
        registry.complete_epoch(
            epoch.id,
            EpochSelection(challenger.id, seeded.baseline.id),
        )

    runtime_cli.main(
        [
            "list-epochs",
            "--config",
            str(config),
            "--lineage",
            str(seeded.lineage_id),
        ]
    )
    value = json.loads(capsys.readouterr().out)
    item = value["epochs"][0]
    assert item["active_agent_version"] == "agent-v0"
    assert item["challenger_agent_versions"] == ["agent-v1"]
    assert item["winner_agent_version"] == "agent-v1"
    assert item["winner_branch"] == "challenger"
    assert item["winner_challenger_ordinal"] == 1
    assert item["decision"] == "challenger_promoted"

    runtime_cli.main(
        [
            "list-epochs",
            "--config",
            str(config),
            "--lineage",
            str(seeded.lineage_id),
            "--format",
            "table",
        ]
    )
    table = capsys.readouterr().out
    assert "ACTIVE_BEFORE" in table
    assert "agent-v0" in table
    assert "agent-v1" in table
    assert "challenger_promoted" in table


def test_attempt_history_cli_keeps_completed_pivot_without_kernel_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _server_only_config(tmp_path)
    settings = RuntimeSettings.from_file(config)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        seeded = seed_lineage(
            registry,
            challenger_count=0,
            attempts_per_trajectory=1,
        )
        lineage = registry.get_lineage(seeded.lineage_id)
        epoch = Epoch(
            id=new_epoch_id(),
            lineage_id=lineage.id,
            number=1,
            active_kernel_agent_revision_id=seeded.active_revision_id,
            challenger_kernel_agent_revision_ids=(),
            starting_kernel_revision_id=seeded.baseline.id,
            evidence_checkpoint=lineage.evidence_checkpoint,
            challenger_count=0,
            trajectories_per_branch=1,
            attempts_per_trajectory=1,
            status=EpochStatus.RUNNING,
            winner_kernel_agent_revision_id=None,
            best_kernel_revision_id=None,
            created_at=NOW,
            completed_at=None,
        )
        registry.insert_epoch(epoch)
        attempt = Attempt(
            id=new_attempt_id(),
            epoch_id=epoch.id,
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=1,
            kernel_agent_revision_id=seeded.active_revision_id,
            input_kernel_revision_id=seeded.baseline.id,
            attempt_evidence_digest=digest("pivot-evidence"),
            output_kernel_revision_id=None,
            accepted_as_branch_best=False,
            status=AttemptStatus.RUNNING,
            infrastructure_failures=0,
            recovery_generation=0,
            authority_started_at=NOW,
            failure_reason=None,
            created_at=NOW,
            completed_at=None,
        )
        registry.insert_attempt(attempt)
        registry.record_attempt_report(
            attempt.id,
            digest("pivot-report"),
            AttemptReportStatus.PIVOT,
        )
        registry.complete_attempt(
            attempt.id,
            None,
            accepted_as_branch_best=False,
            failure_reason="Optimizer ended with pivot",
        )

    args = ["--config", str(config), "--lineage", str(seeded.lineage_id)]
    runtime_cli.main(["list-attempts", *args])
    listing = json.loads(capsys.readouterr().out)
    assert listing["attempts"][0]["attempt_id"] == attempt.id
    assert listing["attempts"][0]["disposition"] == "pivot"
    assert listing["attempts"][0]["candidate_produced"] is False
    assert listing["attempts"][0]["output_kernel_version"] is None

    runtime_cli.main(["list-attempts", *args, "--format", "table"])
    table = capsys.readouterr().out
    assert "ATTEMPT" in table
    assert "pivot" in table
    assert "v0" in table

    runtime_cli.main(["show-attempt", "--config", str(config), "--attempt", str(attempt.id)])
    shown = json.loads(capsys.readouterr().out)
    assert shown["status"] == "completed"
    assert shown["attempt_report_status"] == "pivot"


def test_artifact_gc_apply_requires_explicit_stopped_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirm-runtime-stopped"):
        runtime_cli.main(
            [
                "gc-artifacts",
                "--config",
                str(_server_only_config(tmp_path)),
                "--minimum-age-seconds",
                "3600",
                "--limit",
                "10",
                "--apply",
            ]
        )


def test_workspace_gc_apply_requires_explicit_stopped_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirm-runtime-stopped"):
        runtime_cli.main(
            [
                "gc-workspaces",
                "--config",
                str(_server_only_config(tmp_path)),
                "--minimum-age-seconds",
                "3600",
                "--limit",
                "10",
                "--apply",
            ]
        )


def test_bootstrap_run_cli_lists_and_shows_exact_generation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _server_only_config(tmp_path)
    settings = RuntimeSettings.from_file(config_path)
    signing_key = b"q" * 32
    monkeypatch.setenv("UNUSED_SIGNING_KEY", base64.b64encode(signing_key).decode())
    now = datetime.fromisoformat(NOW)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        control = SqliteGatewayControl(
            settings.storage.gateway_database,
            registry,
            signing_key=signing_key,
            clock=lambda: now,
        )
        attempt_id = new_attempt_id()
        subject = BootstrapGatewaySubject(
            attempt_id=attempt_id,
            campaign_id=new_campaign_id(),
            lineage_id=new_lineage_id(),
            epoch_id=new_epoch_id(),
            kernel_agent_revision_id=new_kernel_agent_revision_id(),
            operator="vector_add",
            hardware_target="nvidia-h100",
            dsl=Dsl.TRITON,
            evaluation_contract_digest=digest("cli-contract"),
            input_kernel_digest=digest("cli-kernel"),
            evidence_digest=digest("cli-evidence"),
            created_at=now,
        )
        capability = control.issue_bootstrap(
            subject,
            GatewayCapabilityPolicy(
                frozenset({GatewayOperation.EVALUATE}),
                1,
                now + timedelta(hours=1),
            ),
        )
        control.begin_bootstrap_run(
            attempt_id,
            capability.recovery_generation,
            run_id="run-cli",
            workspace_path="/runtime/bootstrap/run-cli",
        )
        control.finish_bootstrap_run(
            attempt_id,
            capability.recovery_generation,
            status=BootstrapRunStatus.FAILED,
            finish_reason="blocked",
            failure_reason="Gateway unavailable",
            token_budget=100,
            token_usage=TokenUsage(10, 2, 20, 0),
        )
        source = tmp_path / "evaluated-candidate"
        source.mkdir()
        (source / "kernel.py").write_text("def kernel(): return 1\n")
        artifacts = LocalArtifactStore(settings.storage.artifacts_root)
        candidate = artifacts.put_directory(source, ArtifactKind.KERNEL)
        raw_result = artifacts.put_json(
            {"status": "succeeded", "latency_us": 7.0},
            ArtifactKind.GATEWAY_RESULT,
        )
        evaluation = control.record_evaluation(
            attempt_id,
            source=GatewayEvaluationSource.AGENT,
            idempotency_key="cli-evaluation",
            candidate_artifact_digest=candidate,
            gateway_result_digest=raw_result,
            correct=True,
            latency_us=7.0,
            agate_job_id="ev_cli",
        )
        control.close()

    common = ["--config", str(config_path), "--attempt", str(attempt_id)]
    runtime_cli.main(["list-bootstrap-runs", *common])
    listing = json.loads(capsys.readouterr().out)
    assert listing["runs"][0]["run_id"] == "run-cli"
    assert listing["runs"][0]["token_usage"]["total_tokens"] == 32

    runtime_cli.main(["show-bootstrap-run", *common, "--generation", "0"])
    exact = json.loads(capsys.readouterr().out)
    assert exact["finish_reason"] == "blocked"

    runtime_cli.main(["list-evaluations", *common])
    evaluations = json.loads(capsys.readouterr().out)
    assert evaluations["evaluations"][0]["evaluation_id"] == evaluation.id
    assert evaluations["evaluations"][0]["evaluation_label"] == "g0-e1"

    runtime_cli.main(
        [
            "show-evaluation",
            "--config",
            str(config_path),
            "--evaluation",
            evaluation.id,
            "--source",
            "--result",
        ]
    )
    shown = json.loads(capsys.readouterr().out)
    assert shown["source_files"][0]["content"] == "def kernel(): return 1\n"
    assert shown["result"]["latency_us"] == 7.0


def test_recover_epoch_cli_is_idempotent_and_reports_recovery(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _server_only_config(tmp_path)
    settings = RuntimeSettings.from_file(config_path)
    with SqliteRegistry(settings.storage.registry_database) as registry:
        seeded = seed_lineage(
            registry,
            challenger_count=0,
            attempts_per_trajectory=1,
        )
        epoch = Epoch(
            id=new_epoch_id(),
            lineage_id=seeded.lineage_id,
            number=1,
            active_kernel_agent_revision_id=seeded.active_revision_id,
            challenger_kernel_agent_revision_ids=(),
            starting_kernel_revision_id=seeded.baseline.id,
            evidence_checkpoint=registry.get_lineage(seeded.lineage_id).evidence_checkpoint,
            challenger_count=0,
            trajectories_per_branch=1,
            attempts_per_trajectory=1,
            status=EpochStatus.RUNNING,
            winner_kernel_agent_revision_id=None,
            best_kernel_revision_id=None,
            created_at=NOW,
            completed_at=None,
        )
        registry.insert_epoch(epoch)
        attempt = Attempt(
            id=new_attempt_id(),
            epoch_id=epoch.id,
            branch=BranchRole.ACTIVE,
            challenger_ordinal=0,
            trajectory_ordinal=1,
            ordinal=1,
            kernel_agent_revision_id=seeded.active_revision_id,
            input_kernel_revision_id=seeded.baseline.id,
            attempt_evidence_digest=digest("attempt-evidence"),
            output_kernel_revision_id=None,
            accepted_as_branch_best=False,
            status=AttemptStatus.RUNNING,
            infrastructure_failures=0,
            recovery_generation=0,
            authority_started_at=NOW,
            failure_reason=None,
            created_at=NOW,
            completed_at=None,
        )
        registry.insert_attempt(attempt)
        registry.record_infrastructure_failure(attempt.id, "worker unavailable")
        registry.fail_epoch(epoch.id, "retry budget exhausted")

    argv = [
        "recover-epoch",
        "--config",
        str(config_path),
        "--epoch",
        str(epoch.id),
        "--recovery-key",
        "ticket-123",
        "--reason",
        "worker host replaced",
    ]
    runtime_cli.main(argv)
    first = json.loads(capsys.readouterr().out)
    runtime_cli.main(argv)
    second = json.loads(capsys.readouterr().out)

    assert second == first
    assert first["epoch_id"] == epoch.id
    assert first["attempt_ids"] == [attempt.id]
    assert first["generation"] == 1
    with SqliteRegistry(settings.storage.registry_database) as registry:
        assert registry.get_attempt(attempt.id).recovery_generation == 1


def test_drain_wiki_feedback_cli_reports_one_batch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drainer = RecordingDrainer()
    runtime = FakeWikiFeedbackRuntime(drainer)

    def build_fake(
        settings: RuntimeSettings,
        environment: object,
    ) -> FakeWikiFeedbackRuntime:
        del settings, environment
        return runtime

    monkeypatch.setattr(runtime_cli_maintenance, "build_wiki_feedback_runtime", build_fake)

    runtime_cli.main(
        [
            "drain-wiki-feedback",
            "--config",
            str(_server_only_config(tmp_path)),
        ]
    )

    assert drainer.called
    assert runtime.closed
    assert json.loads(capsys.readouterr().out) == {
        "claimed": 3,
        "completed": 2,
        "retried": 1,
        "failed": 0,
    }


def test_wiki_feedback_administration_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeWikiFeedbackRuntime(RecordingDrainer())

    def build_fake(settings: RuntimeSettings, environment: object) -> FakeWikiFeedbackRuntime:
        del settings, environment
        return runtime

    monkeypatch.setattr(runtime_cli_maintenance, "build_wiki_feedback_runtime", build_fake)
    item_id = new_wiki_feedback_id()

    runtime_cli.main(
        [
            "requeue-wiki-feedback",
            "--config",
            str(_server_only_config(tmp_path)),
            "--item",
            str(item_id),
        ]
    )
    assert json.loads(capsys.readouterr().out) == {
        "item_id": item_id,
        "status": "pending",
    }
    runtime.closed = False
    runtime_cli.main(
        [
            "maintain-wiki-feedback",
            "--config",
            str(_server_only_config(tmp_path)),
            "--compact",
        ]
    )
    assert json.loads(capsys.readouterr().out) == {"compacted": True, "pruned": 7}


def test_worker_session_cli_lists_and_shows_raw_trace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _server_only_config(tmp_path)
    settings = RuntimeSettings.from_file(config)
    with SqliteRegistry(settings.storage.registry_database, clock=lambda: NOW) as registry:
        session = registry.start_worker_session(
            WorkerSession(
                id=new_worker_session_id(),
                role=WorkerSessionRole.PROBLEM_GENERALIZATION,
                subject_id="cli-generalization",
                external_run_id="cli-run",
                workspace_path="/runtime/cli-run",
                status=WorkerSessionStatus.RUNNING,
                started_at=NOW,
                backend="codex",
            )
        )
        registry.finish_worker_session(
            session.id,
            status=WorkerSessionStatus.COMPLETED,
            finish_reason="completed",
            trace_digest=digest("cli-session-trace"),
            token_budget=1000,
            token_usage=TokenUsage(10, 2, 3, 0),
        )

    runtime_cli.main(
        [
            "list-worker-sessions",
            "--config",
            str(config),
            "--subject",
            "cli-generalization",
        ]
    )
    listing = json.loads(capsys.readouterr().out)
    assert listing["worker_sessions"][0]["worker_session_id"] == session.id
    assert listing["worker_sessions"][0]["session_trace_digest"] == digest("cli-session-trace")

    runtime_cli.main(
        [
            "show-worker-session",
            "--config",
            str(config),
            "--session",
            str(session.id),
        ]
    )
    exact = json.loads(capsys.readouterr().out)
    assert exact["status"] == "completed"
    assert exact["token_usage"]["total_tokens"] == 15
