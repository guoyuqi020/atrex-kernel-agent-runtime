"""Recoverable single- and multi-lineage Campaign scheduling tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import pytest
from conftest import NOW, FakeAttemptEvidence, digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.controller import (
    CampaignScheduler,
    CampaignScheduleResult,
    EpochController,
    EvidenceCheckpointV1,
    LocalEvidenceAssembler,
    RegistryLineageLeaseManager,
)
from atrex_runtime.domain.errors import LineageLeaseUnavailableError
from atrex_runtime.domain.ids import (
    CampaignId,
    LineageId,
    new_campaign_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
)
from atrex_runtime.domain.models import (
    Campaign,
    CampaignStatus,
    Dsl,
    KernelAgentRevision,
    KernelEvaluation,
    KernelRevision,
    Lineage,
    LineageStatus,
)
from atrex_runtime.ports import (
    AttemptCandidateResult,
    BuildChallengerRequest,
    BuildChallengerResult,
    KernelAgentCandidate,
    KernelAgentCandidateProposal,
    RunAgentWorkflowRequest,
    RunAttemptRequest,
    RunAttemptResult,
)
from atrex_runtime.registry.sqlite import SqliteRegistry


@dataclass
class AdvancingEvolver:
    """Create one distinct fixed-Evolver Challenger for every epoch."""

    calls: list[BuildChallengerRequest] = field(default_factory=list)

    async def build_challenger(self, request: BuildChallengerRequest) -> BuildChallengerResult:
        self.calls.append(request)
        ordinal = len(self.calls)
        return BuildChallengerResult(
            KernelAgentCandidateProposal(
                "evolved",
                request.parent_revision.id,
                KernelAgentCandidate(
                    dsl=request.parent_revision.dsl,
                    optimizer_digest=digest(f"challenger-optimizer-{ordinal}"),
                    runtime_state_digest=digest(f"challenger-runtime-state-{ordinal}"),
                ),
            ),
            digest(f"evolution-trace-{ordinal}"),
        )


@dataclass
class ImprovingOptimizer:
    """Return a globally decreasing authoritative latency for every Attempt."""

    calls: list[RunAttemptRequest] = field(default_factory=list)

    async def run_attempt(self, request: RunAttemptRequest) -> RunAttemptResult:
        self.calls.append(request)
        ordinal = len(self.calls)
        return RunAttemptResult(
            candidate=AttemptCandidateResult(
                artifact_digest=digest(f"candidate-kernel-{ordinal}"),
                gateway_result_digest=digest(f"candidate-gateway-{ordinal}"),
                correct=True,
                latency_us=100.0 - ordinal,
            )
        )


def test_completed_bootstrap_evidence_contains_only_report_and_conversation(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    report_digest = artifacts.put_json(
        {"status": "baseline_ready", "approach": "plain baseline"},
        ArtifactKind.ATTEMPT_REPORT,
    )
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "conversation.jsonl").write_text(
        json.dumps({"type": "assistant/message", "text": "baseline complete"}) + "\n",
        encoding="utf-8",
    )
    session_digest = artifacts.put_directory(trace, ArtifactKind.SESSION_LOG)
    lineage_id = new_lineage_id()
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        checkpoint = LocalEvidenceAssembler(registry, artifacts).create_bootstrap(
            lineage_id,
            report_digest=report_digest,
            session_trace_digest=session_digest,
        )

    payload = artifacts.verify(checkpoint).payload_path
    assert {path.name for path in payload.iterdir()} == {"bootstrap", "checkpoint.json"}
    assert {path.name for path in (payload / "bootstrap").iterdir()} == {
        "report.json",
        "conversation.jsonl",
    }
    assert json.loads((payload / "bootstrap/report.json").read_text()) == {
        "status": "baseline_ready",
        "approach": "plain baseline",
    }
    assert "baseline complete" in (payload / "bootstrap/conversation.jsonl").read_text()


def _seed_lineage(
    registry: SqliteRegistry,
    artifacts: LocalArtifactStore,
    tmp_path: Path,
    campaign_id: CampaignId,
    dsl: Dsl,
    *,
    name: str | None = None,
    max_challengers: int = 1,
    observer_lineage_id: LineageId | None = None,
) -> LineageId:
    lineage_id = new_lineage_id()
    identity = dsl.value if name is None else name
    initial = tmp_path / f"initial-{identity}"
    initial.mkdir()
    (initial / "baseline.md").write_text(f"{dsl.value} baseline\n", encoding="utf-8")
    evidence = LocalEvidenceAssembler(registry, artifacts).create_initial(lineage_id, initial)
    agent_id = new_kernel_agent_revision_id()
    registry.register_kernel_agent_revision(
        KernelAgentRevision(
            id=agent_id,
            parent_id=None,
            creation_key=f"bootstrap:{identity}",
            dsl=dsl,
            optimizer_digest=digest(f"{identity}-optimizer"),
            created_by="bootstrap",
            created_at=NOW,
            source_provenance_digest=digest(f"{identity}-source"),
        )
    )
    kernel_id = new_kernel_revision_id()
    registry.register_kernel_revision(
        KernelRevision(
            kernel_id,
            None,
            digest(f"{identity}-baseline-kernel"),
            None,
            KernelEvaluation(True, 100.0, digest(f"{identity}-baseline-gateway")),
            NOW,
        )
    )
    registry.insert_lineage(
        Lineage(
            id=lineage_id,
            campaign_id=campaign_id,
            dsl=dsl,
            hardware_target="nvidia-h100",
            active_kernel_agent_revision_id=agent_id,
            best_kernel_revision_id=kernel_id,
            evidence_checkpoint=evidence,
            max_challengers=max_challengers,
            optimizer_attempt_budget=2,
            next_epoch_number=1,
            status=LineageStatus.READY,
            evolver_observer_lineage_id=observer_lineage_id,
        )
    )
    return lineage_id


class SchedulerWorkflow:
    """Explicit two-Branch Workflow used by scheduler integration tests."""

    async def run(self, request: RunAgentWorkflowRequest, operations: object) -> None:
        execute = operations.execute_workflow_operation  # type: ignore[attr-defined]

        async def call(operation: str, arguments: Mapping[str, object]):
            return await execute(
                operation,
                {**arguments, "_runtime_workflow_program_sha256": "a" * 64},
            )

        operation = "replicate_active" if request.epoch_number == 1 else "evolve_agent"
        await call(operation, {"challenger_ordinal": 1})
        for branch in ("active", "challenger-1"):
            await call(
                "create_trajectory",
                {
                    "branch": branch,
                    "trajectory_ordinal": 1,
                    "trajectory_count": 1,
                    "attempt_capacity": 1,
                },
            )
        await call(
            "run_attempts_parallel",
            {
                "launches": [
                    {
                        "branch": branch,
                        "trajectory_ordinal": 1,
                        "attempt_ordinal": 1,
                    }
                    for branch in ("active", "challenger-1")
                ]
            },
        )
        kernel = await call("select_best_kernel", {})
        agent = await call("compare_agents", {})
        await call(
            "complete_epoch",
            {
                "kernel_revision_id": kernel["kernel_revision_id"],
                "kernel_agent_revision_id": agent["kernel_agent_revision_id"],
            },
        )


class PairedLineageWorkflow:
    """Run Active as Isolated and Challenger as isolated evolution."""

    async def run(self, request: RunAgentWorkflowRequest, operations: object) -> None:
        execute = operations.execute_workflow_operation  # type: ignore[attr-defined]

        async def call(operation: str, arguments: Mapping[str, object]):
            return await execute(
                operation,
                {**arguments, "_runtime_workflow_program_sha256": "b" * 64},
            )

        branch = "active"
        if request.max_challengers:
            branch = "challenger-1"
            await call(
                "replicate_active" if request.epoch_number == 1 else "evolve_agent",
                {"challenger_ordinal": 1},
            )
        await call(
            "create_trajectory",
            {
                "branch": branch,
                "trajectory_ordinal": 1,
                "trajectory_count": 1,
                "attempt_capacity": request.optimizer_attempt_budget,
            },
        )
        for attempt_ordinal in range(1, request.optimizer_attempt_budget + 1):
            await call(
                "run_attempts_parallel",
                {
                    "launches": [
                        {
                            "branch": branch,
                            "trajectory_ordinal": 1,
                            "attempt_ordinal": attempt_ordinal,
                        }
                    ]
                },
            )
        kernel = await call("select_best_kernel", {})
        agent = await call("compare_agents", {})
        await call(
            "complete_epoch",
            {
                "kernel_revision_id": kernel["kernel_revision_id"],
                "kernel_agent_revision_id": agent["kernel_agent_revision_id"],
            },
        )


class PostEpochEvolutionWorkflow:
    """Run Active, then request a successor after this Epoch's Evidence is sealed."""

    async def run(self, request: RunAgentWorkflowRequest, operations: object) -> None:
        execute = operations.execute_workflow_operation  # type: ignore[attr-defined]

        async def call(operation: str, arguments: Mapping[str, object]):
            return await execute(
                operation,
                {**arguments, "_runtime_workflow_program_sha256": "c" * 64},
            )

        await call(
            "create_trajectory",
            {
                "branch": "active",
                "trajectory_ordinal": 1,
                "trajectory_count": 1,
                "attempt_capacity": request.optimizer_attempt_budget,
            },
        )
        for attempt_ordinal in range(1, request.optimizer_attempt_budget + 1):
            await call(
                "run_attempts_parallel",
                {
                    "launches": [
                        {
                            "branch": "active",
                            "trajectory_ordinal": 1,
                            "attempt_ordinal": attempt_ordinal,
                        }
                    ]
                },
            )
        if request.max_challengers:
            scheduled = await call("evolve_agent", {"challenger_ordinal": 1})
            assert scheduled["scheduled_after_epoch"] is True
        kernel = await call("select_best_kernel", {})
        agent = await call("compare_agents", {})
        await call(
            "complete_epoch",
            {
                "kernel_revision_id": kernel["kernel_revision_id"],
                "kernel_agent_revision_id": agent["kernel_agent_revision_id"],
            },
        )


def _scheduler(
    registry: SqliteRegistry,
    artifacts: LocalArtifactStore,
    lease_root: Path,
) -> tuple[CampaignScheduler, EpochController, AdvancingEvolver, ImprovingOptimizer]:
    evolver = AdvancingEvolver()
    optimizer = ImprovingOptimizer()
    controller = EpochController(
        registry,
        evolver,
        optimizer,
        FakeAttemptEvidence(),
        workflow_runner=SchedulerWorkflow(),
    )
    scheduler = CampaignScheduler(
        registry,
        controller,
        LocalEvidenceAssembler(registry, artifacts),
        RegistryLineageLeaseManager(
            registry,
            lease_seconds=10,
            heartbeat_seconds=1,
        ),
    )
    return scheduler, controller, evolver, optimizer


@pytest.mark.anyio
async def test_scheduler_follows_workflow_owned_evolution_timing(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        campaign_id = new_campaign_id()
        registry.insert_campaign(
            Campaign(
                campaign_id,
                "vector_add",
                "nvidia-h100",
                digest("contract"),
                digest("problem"),
                NOW,
            )
        )
        lineage_id = _seed_lineage(registry, artifacts, tmp_path, campaign_id, Dsl.TRITON)
        scheduler, _controller, evolver, optimizer = _scheduler(
            registry,
            artifacts,
            tmp_path / "leases",
        )

        result = await scheduler.run_lineage_through(lineage_id, 3)

        assert result.completed_epochs == (1, 2, 3)
        assert len(evolver.calls) == 2
        assert len(optimizer.calls) == 6
        epochs = registry.list_epochs(lineage_id)
        assert [epoch.max_challengers for epoch in epochs] == [1, 1, 1]
        assert [len(registry.list_attempts(epoch.id)) for epoch in epochs] == [2, 2, 2]


@pytest.mark.anyio
async def test_scheduler_evolves_successor_from_completed_epoch_evidence(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        campaign_id = new_campaign_id()
        registry.insert_campaign(
            Campaign(
                campaign_id,
                "vector_add",
                "nvidia-h100",
                digest("contract"),
                digest("problem"),
                NOW,
            )
        )
        lineage_id = _seed_lineage(registry, artifacts, tmp_path, campaign_id, Dsl.TRITON)
        evolver = AdvancingEvolver()
        optimizer = ImprovingOptimizer()
        controller = EpochController(
            registry,
            evolver,
            optimizer,
            FakeAttemptEvidence(),
            workflow_runner=PostEpochEvolutionWorkflow(),
        )
        scheduler = CampaignScheduler(
            registry,
            controller,
            LocalEvidenceAssembler(registry, artifacts),
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=10,
                heartbeat_seconds=1,
            ),
        )

        result = await scheduler.run_lineage_through(lineage_id, 2)

        assert result.completed_epochs == (1, 2)
        assert [request.epoch_number for request in evolver.calls] == [2, 3]
        epochs = registry.list_epochs(lineage_id)
        assert epochs[1].active_kernel_agent_revision_id != (
            epochs[0].active_kernel_agent_revision_id
        )
        assert result.lineage.active_kernel_agent_revision_id != (
            epochs[1].active_kernel_agent_revision_id
        )
        for epoch, request in zip(epochs, evolver.calls, strict=True):
            successor = registry.get_epoch_successor_evolution(epoch.id)
            assert successor is not None
            assert successor.status == "completed"
            checkpoint = EvidenceCheckpointV1.from_file(
                artifacts.verify(request.evidence_checkpoint).payload_path
                / "checkpoint.json"
            )
            assert checkpoint.through_epoch == epoch.number

        resumed = await scheduler.run_lineage_through(lineage_id, 2)
        assert resumed.completed_epochs == ()
        assert len(evolver.calls) == 2


@pytest.mark.anyio
async def test_scheduler_runs_multiple_dsl_lineages_through_target(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        campaign_id = new_campaign_id()
        registry.insert_campaign(
            Campaign(
                campaign_id,
                "vector_add",
                "nvidia-h100",
                digest("contract"),
                digest("problem"),
                NOW,
            )
        )
        triton = _seed_lineage(registry, artifacts, tmp_path, campaign_id, Dsl.TRITON)
        cuda = _seed_lineage(registry, artifacts, tmp_path, campaign_id, Dsl.CUDA)
        scheduler, _controller, evolver, optimizer = _scheduler(
            registry,
            artifacts,
            tmp_path / "leases",
        )

        result = await scheduler.run_campaign_through((triton, cuda), 2)

        assert result.campaign_id == campaign_id
        assert [item.lineage.id for item in result.lineages] == [triton, cuda]
        assert all(item.completed_epochs == (1, 2) for item in result.lineages)
        assert len(evolver.calls) == 2
        assert len(optimizer.calls) == 8
        for lineage_id in (triton, cuda):
            lineage = registry.get_lineage(lineage_id)
            assert lineage.status is LineageStatus.READY
            assert lineage.next_epoch_number == 3
            stored = artifacts.verify(lineage.evidence_checkpoint)
            checkpoint = EvidenceCheckpointV1.from_file(stored.payload_path / "checkpoint.json")
            assert checkpoint.through_epoch == 2
            assert sorted(path.name for path in (stored.payload_path / "epochs").iterdir()) == [
                "00000001.json",
                "00000002.json",
            ]


@pytest.mark.anyio
async def test_external_isolated_observer_blocks_evolution_until_preceding_epoch(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        active_campaign_id = new_campaign_id()
        challenger_campaign_id = new_campaign_id()
        for campaign_id in (active_campaign_id, challenger_campaign_id):
            registry.insert_campaign(
                Campaign(
                    campaign_id,
                    "vector_add",
                    "nvidia-h100",
                    digest("contract"),
                    digest("problem"),
                    NOW,
                )
            )
        active = _seed_lineage(
            registry,
            artifacts,
            tmp_path,
            active_campaign_id,
            Dsl.TRITON,
            name="paired-active",
            max_challengers=0,
        )
        challenger = _seed_lineage(
            registry,
            artifacts,
            tmp_path,
            challenger_campaign_id,
            Dsl.TRITON,
            name="paired-challenger",
            max_challengers=1,
            observer_lineage_id=active,
        )
        evolver = AdvancingEvolver()
        optimizer = ImprovingOptimizer()
        controller = EpochController(
            registry,
            evolver,
            optimizer,
            FakeAttemptEvidence(),
            workflow_runner=PairedLineageWorkflow(),
        )
        scheduler = CampaignScheduler(
            registry,
            controller,
            LocalEvidenceAssembler(registry, artifacts),
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=10,
                heartbeat_seconds=1,
            ),
            observer_poll_seconds=0.001,
        )
        results: dict[str, CampaignScheduleResult] = {}

        async def run_challenger() -> None:
            results["challenger"] = await scheduler.run_campaign_through((challenger,), 2)

        async def run_active() -> None:
            results["active"] = await scheduler.run_campaign_through((active,), 2)

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run_challenger)
            await anyio.sleep(0.01)
            # Epoch 1 may finish, but Evolution for Epoch 2 must await the external
            # Isolated control's published Epoch-1 checkpoint.
            assert evolver.calls == []
            tasks.start_soon(run_active)

        assert results["active"].campaign_id == active_campaign_id
        assert results["challenger"].campaign_id == challenger_campaign_id
        assert results["active"].lineages[0].completed_epochs == (1, 2)
        assert results["challenger"].lineages[0].completed_epochs == (1, 2)
        assert len(evolver.calls) == 1
        request = evolver.calls[0]
        assert request.epoch_number == 2
        assert len(request.references) == 1
        reference = request.references[0]
        assert reference.name == "control"
        assert reference.lineage_id == active
        checkpoint = EvidenceCheckpointV1.from_file(
            artifacts.verify(reference.evidence_checkpoint).payload_path
            / "checkpoint.json"
        )
        assert checkpoint.lineage_id == active
        assert checkpoint.through_epoch == 1
        assert {entry.lineage_id for entry in reference.agent_catalog} == {active}
        assert all(
            entry.epoch_number is None or entry.epoch_number < request.epoch_number
            for entry in reference.kernel_catalog
        )
        assert any(entry.epoch_number == 1 for entry in reference.kernel_catalog)
        assert {entry.lineage_id for entry in request.agent_catalog} == {challenger}
        assert len(registry.list_attempts(registry.find_epoch(active, 1).id)) == 2  # type: ignore[union-attr]
        assert len(registry.list_attempts(registry.find_epoch(challenger, 1).id)) == 2  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_post_epoch_evolution_waits_for_matching_observer_epoch(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        observer_campaign_id = new_campaign_id()
        evolved_campaign_id = new_campaign_id()
        for campaign_id in (observer_campaign_id, evolved_campaign_id):
            registry.insert_campaign(
                Campaign(
                    campaign_id,
                    "vector_add",
                    "nvidia-h100",
                    digest("contract"),
                    digest("problem"),
                    NOW,
                )
            )
        observer = _seed_lineage(
            registry,
            artifacts,
            tmp_path,
            observer_campaign_id,
            Dsl.TRITON,
            name="post-epoch-observer",
            max_challengers=0,
        )
        evolved = _seed_lineage(
            registry,
            artifacts,
            tmp_path,
            evolved_campaign_id,
            Dsl.TRITON,
            name="post-epoch-evolved",
            max_challengers=1,
            observer_lineage_id=observer,
        )
        evolver = AdvancingEvolver()
        controller = EpochController(
            registry,
            evolver,
            ImprovingOptimizer(),
            FakeAttemptEvidence(),
            workflow_runner=PostEpochEvolutionWorkflow(),
        )
        scheduler = CampaignScheduler(
            registry,
            controller,
            LocalEvidenceAssembler(registry, artifacts),
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=10,
                heartbeat_seconds=1,
            ),
            observer_poll_seconds=0.001,
        )

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(scheduler.run_campaign_through, (evolved,), 1)
            await anyio.sleep(0.01)
            assert evolver.calls == []
            tasks.start_soon(scheduler.run_campaign_through, (observer,), 1)

        assert len(evolver.calls) == 1
        request = evolver.calls[0]
        assert request.epoch_number == 2
        assert len(request.references) == 1
        reference = request.references[0]
        assert reference.name == "control"
        assert reference.lineage_id == observer
        checkpoint = EvidenceCheckpointV1.from_file(
            artifacts.verify(reference.evidence_checkpoint).payload_path
            / "checkpoint.json"
        )
        assert checkpoint.through_epoch == 1


@pytest.mark.anyio
async def test_scheduler_discovers_and_finalizes_registered_campaign(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        campaign_id = new_campaign_id()
        registry.insert_campaign(
            Campaign(
                campaign_id,
                "vector_add",
                "nvidia-h100",
                digest("contract"),
                digest("problem"),
                NOW,
            )
        )
        triton = _seed_lineage(registry, artifacts, tmp_path, campaign_id, Dsl.TRITON)
        cuda = _seed_lineage(registry, artifacts, tmp_path, campaign_id, Dsl.CUDA)
        scheduler, _controller, _evolver, _optimizer = _scheduler(
            registry,
            artifacts,
            tmp_path / "leases",
        )

        result = await scheduler.run_registered_campaign_through(
            campaign_id,
            1,
            finalize=True,
        )

        assert [item.lineage.id for item in result.lineages] == [cuda, triton]
        assert registry.get_campaign(campaign_id).status is CampaignStatus.COMPLETED
        assert all(
            lineage.status is LineageStatus.COMPLETED
            for lineage in registry.list_campaign_lineages(campaign_id)
        )


def test_registry_cancels_only_quiescent_campaigns(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        campaign_id = new_campaign_id()
        registry.insert_campaign(
            Campaign(
                campaign_id,
                "vector_add",
                "nvidia-h100",
                digest("contract"),
                digest("problem"),
                NOW,
            )
        )
        lineage_id = _seed_lineage(
            registry,
            artifacts,
            tmp_path,
            campaign_id,
            Dsl.TRITON,
        )

        cancelled = registry.cancel_campaign(campaign_id)

        assert cancelled.status is CampaignStatus.CANCELLED
        assert registry.get_lineage(lineage_id).status is LineageStatus.CANCELLED


@pytest.mark.anyio
async def test_scheduler_recovers_evidence_handoff_without_repeating_epoch(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        campaign_id = new_campaign_id()
        registry.insert_campaign(
            Campaign(
                campaign_id,
                "vector_add",
                "nvidia-h100",
                digest("contract"),
                digest("problem"),
                NOW,
            )
        )
        lineage_id = _seed_lineage(registry, artifacts, tmp_path, campaign_id, Dsl.TRITON)
        scheduler, controller, _evolver, optimizer = _scheduler(
            registry,
            artifacts,
            tmp_path / "leases",
        )
        await controller.run_epoch(lineage_id, 1)
        calls_after_epoch = len(optimizer.calls)
        assert registry.get_lineage(lineage_id).status is LineageStatus.AWAITING_EVIDENCE
        assembler = LocalEvidenceAssembler(registry, artifacts)
        assert assembler.assemble_next(lineage_id) == assembler.assemble_next(lineage_id)

        result = await scheduler.run_lineage_through(lineage_id, 1)

        assert result.completed_epochs == ()
        assert len(optimizer.calls) == calls_after_epoch
        assert result.lineage.status is LineageStatus.READY
        assert result.lineage.next_epoch_number == 2


@pytest.mark.anyio
async def test_scheduler_refuses_a_lineage_owned_by_another_process(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    lease_root = tmp_path / "leases"
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        campaign_id = new_campaign_id()
        registry.insert_campaign(
            Campaign(
                campaign_id,
                "vector_add",
                "nvidia-h100",
                digest("contract"),
                digest("problem"),
                NOW,
            )
        )
        lineage_id = _seed_lineage(
            registry,
            artifacts,
            tmp_path,
            campaign_id,
            Dsl.TRITON,
        )
        scheduler, _controller, evolver, optimizer = _scheduler(
            registry,
            artifacts,
            lease_root,
        )

        with (
            RegistryLineageLeaseManager(
                registry,
                lease_seconds=10,
                heartbeat_seconds=1,
            ).acquire(lineage_id),
            pytest.raises(LineageLeaseUnavailableError),
        ):
            await scheduler.run_lineage_through(lineage_id, 1)

        assert evolver.calls == []
        assert optimizer.calls == []
        result = await scheduler.run_lineage_through(lineage_id, 1)
        assert result.completed_epochs == (1,)
