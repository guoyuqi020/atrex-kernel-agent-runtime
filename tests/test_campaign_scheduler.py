"""Recoverable single- and multi-lineage Campaign scheduling tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import NOW, FakeAttemptEvidence, digest

from atrex_runtime.artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from atrex_runtime.controller import (
    CampaignScheduler,
    EpochController,
    EvidenceCheckpointV1,
    LocalEvidenceAssembler,
    RegistryLineageLeaseManager,
)
from atrex_runtime.controller.projection import (
    EvidenceArtifactProjector,
    EvidenceProjectionLimits,
)
from atrex_runtime.domain.errors import LineageLeaseUnavailableError
from atrex_runtime.domain.ids import (
    ArtifactDigest,
    AttemptId,
    CampaignId,
    LineageId,
    new_campaign_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
    parse_artifact_digest,
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
    WikiFeedbackStatus,
)
from atrex_runtime.gateway.control import GatewayOperation
from atrex_runtime.knowledge import (
    KnowledgeInteractionV1,
    KnowledgeQueryV1,
    KnowledgeSnapshotResponseV1,
    LocalWikiFeedbackPreparer,
    WikiFeedbackPreparer,
    WikiFeedbackReportV1,
)
from atrex_runtime.knowledge.models import canonical_json_bytes
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


@dataclass
class SynthesizedWikiInteractionSource:
    """Create one frozen interaction for each completed fake Optimizer Attempt."""

    registry: SqliteRegistry
    artifacts: LocalArtifactStore

    def list_operation_artifacts(
        self,
        attempt_ids: tuple[AttemptId, ...],
        operation: GatewayOperation,
    ) -> tuple[tuple[AttemptId, str, ArtifactDigest], ...]:
        assert operation is GatewayOperation.WIKI_QUERY
        content: JsonValue = {"recommendations": ["tile by 128"]}
        content_digest = parse_artifact_digest(
            f"sha256:{hashlib.sha256(canonical_json_bytes(content)).hexdigest()}"
        )
        rows: list[tuple[AttemptId, str, ArtifactDigest]] = []
        for attempt_id in attempt_ids:
            attempt = self.registry.get_attempt(attempt_id)
            epoch = self.registry.get_epoch(attempt.epoch_id)
            lineage = self.registry.get_lineage(epoch.lineage_id)
            campaign = self.registry.get_campaign(lineage.campaign_id)
            query = KnowledgeQueryV1(
                campaign_id=campaign.id,
                lineage_id=lineage.id,
                epoch_id=epoch.id,
                epoch_number=epoch.number,
                attempt_id=attempt.id,
                branch=attempt.branch,
                attempt_ordinal=attempt.ordinal,
                kernel_agent_revision_id=attempt.kernel_agent_revision_id,
                operator=campaign.operator,
                dsl=lineage.dsl,
                hardware_target=lineage.hardware_target,
                evaluation_contract_digest=campaign.evaluation_contract_digest,
                epoch_evidence_checkpoint_digest=epoch.evidence_checkpoint,
                attempt_evidence_digest=attempt.attempt_evidence_digest,
                query="Which tile should this Attempt try?",
            )
            response = KnowledgeSnapshotResponseV1(
                snapshot_id=f"snapshot-{operation.value}-{attempt_id}",
                content_digest=content_digest,
                content=content,
            )
            idempotency_key = f"{operation.value}-{attempt_id}"
            interaction = KnowledgeInteractionV1(
                idempotency_key=idempotency_key,
                query=query,
                response=response,
            )
            artifact_digest = self.artifacts.put_json(
                interaction.model_dump(mode="json"),
                ArtifactKind.WIKI_INTERACTION,
            )
            rows.append((attempt_id, idempotency_key, artifact_digest))
        return tuple(rows)


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
    wiki_feedback: WikiFeedbackPreparer | None = None,
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
        wiki_feedback,
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
async def test_scheduler_atomically_enqueues_sealed_wiki_feedback_report(
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
        lineage_id = _seed_lineage(
            registry,
            artifacts,
            tmp_path,
            campaign_id,
            Dsl.TRITON,
        )
        scheduler, _controller, _evolver, _optimizer = _scheduler(
            registry,
            artifacts,
            tmp_path / "leases",
            wiki_feedback=LocalWikiFeedbackPreparer(
                registry,
                artifacts,
                SynthesizedWikiInteractionSource(registry, artifacts),
                EvidenceArtifactProjector(
                    artifacts,
                    EvidenceProjectionLimits(16, 1_000_000, 1000, 100_000, 32, 1_000_000),
                ),
            ),
        )

        await scheduler.run_lineage_through(lineage_id, 1)

        lineage = registry.get_lineage(lineage_id)
        items = registry.list_wiki_feedback()
        assert len(items) == 1
        item = items[0]
        report_artifact = artifacts.verify(item.report_artifact_digest)
        report = WikiFeedbackReportV1.model_validate_json(
            (report_artifact.payload_path / "value.json").read_bytes()
        )
        first_claim = registry.claim_wiki_feedback(
            "worker-a",
            now="2099-01-01T00:00:00+00:00",
            lease_expires_at="2099-01-01T00:01:00+00:00",
            limit=1,
        )
        assert first_claim[0].attempt_count == 1
        assert (
            registry.claim_wiki_feedback(
                "worker-b",
                now="2099-01-01T00:00:30+00:00",
                lease_expires_at="2099-01-01T00:01:30+00:00",
                limit=1,
            )
            == []
        )
        recovered = registry.claim_wiki_feedback(
            "worker-b",
            now="2099-01-01T00:01:00+00:00",
            lease_expires_at="2099-01-01T00:02:00+00:00",
            limit=1,
        )
        assert recovered[0].attempt_count == 2
        registry.fail_wiki_feedback(
            recovered[0].id,
            "worker-b",
            error="operator-inspected schema rejection",
        )
        requeued = registry.requeue_wiki_feedback(
            recovered[0].id,
            available_at="2099-01-01T00:02:00+00:00",
        )
        final_claim = registry.claim_wiki_feedback(
            "worker-c",
            now="2099-01-01T00:02:00+00:00",
            lease_expires_at="2099-01-01T00:03:00+00:00",
            limit=1,
        )
        registry.complete_wiki_feedback(final_claim[0].id, "worker-c")
        completed_item = registry.list_wiki_feedback()[0]
        pruned = registry.prune_wiki_feedback(
            completed_before="2100-01-01T00:00:00+00:00",
            limit=10,
        )

    assert report_artifact.kind is ArtifactKind.WIKI_FEEDBACK_REPORT
    assert item.lineage_id == lineage_id
    assert item.epoch_number == 1
    assert completed_item.status is WikiFeedbackStatus.COMPLETED
    assert requeued.status is WikiFeedbackStatus.PENDING
    assert requeued.last_error is None
    assert pruned == 1
    assert report.evidence_checkpoint_digest == lineage.evidence_checkpoint
    assert report.campaign_id == campaign_id
    assert report.lineage_id == lineage_id
    assert report.epoch_number == 1
    assert len(report.attempts) == 2
    assert all(len(attempt.interactions) == 1 for attempt in report.attempts)
    assert all(
        isinstance(attempt.interactions[0].interaction, KnowledgeInteractionV1)
        and attempt.interactions[0].interaction.query.attempt_id == attempt.attempt_id
        for attempt in report.attempts
    )


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
