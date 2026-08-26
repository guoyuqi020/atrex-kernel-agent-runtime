"""Recoverable single- and multi-lineage Campaign scheduling tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import NOW, FakeAttemptEvidence, digest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.controller import (
    CampaignScheduler,
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
    challenger_start_epoch: int = 1,
) -> LineageId:
    lineage_id = new_lineage_id()
    initial = tmp_path / f"initial-{dsl.value}"
    initial.mkdir()
    (initial / "baseline.md").write_text(f"{dsl.value} baseline\n", encoding="utf-8")
    evidence = LocalEvidenceAssembler(registry, artifacts).create_initial(lineage_id, initial)
    agent_id = new_kernel_agent_revision_id()
    registry.register_kernel_agent_revision(
        KernelAgentRevision(
            id=agent_id,
            parent_id=None,
            creation_key=f"bootstrap:{dsl.value}",
            dsl=dsl,
            optimizer_digest=digest(f"{dsl.value}-optimizer"),
            created_by="bootstrap",
            created_at=NOW,
            source_provenance_digest=digest(f"{dsl.value}-source"),
        )
    )
    kernel_id = new_kernel_revision_id()
    registry.register_kernel_revision(
        KernelRevision(
            kernel_id,
            None,
            digest(f"{dsl.value}-baseline-kernel"),
            None,
            KernelEvaluation(True, 100.0, digest(f"{dsl.value}-baseline-gateway")),
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
            challenger_count=1,
            challenger_start_epoch=challenger_start_epoch,
            trajectories_per_branch=1,
            attempts_per_trajectory=1,
            next_epoch_number=1,
            status=LineageStatus.READY,
        )
    )
    return lineage_id


def _scheduler(
    registry: SqliteRegistry,
    artifacts: LocalArtifactStore,
    lease_root: Path,
) -> tuple[CampaignScheduler, EpochController, AdvancingEvolver, ImprovingOptimizer]:
    evolver = AdvancingEvolver()
    optimizer = ImprovingOptimizer()
    controller = EpochController(registry, evolver, optimizer, FakeAttemptEvidence())
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
async def test_scheduler_delays_challengers_until_configured_epoch(tmp_path: Path) -> None:
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
            challenger_start_epoch=2,
        )
        scheduler, _controller, evolver, optimizer = _scheduler(
            registry,
            artifacts,
            tmp_path / "leases",
        )

        result = await scheduler.run_lineage_through(lineage_id, 3)

        assert result.completed_epochs == (1, 2, 3)
        assert len(evolver.calls) == 2
        assert len(optimizer.calls) == 5
        epochs = registry.list_epochs(lineage_id)
        assert [epoch.challenger_count for epoch in epochs] == [0, 1, 1]
        assert [len(registry.list_attempts(epoch.id)) for epoch in epochs] == [1, 2, 2]


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
        assert len(evolver.calls) == 4
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
