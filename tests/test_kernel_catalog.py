"""Durable Kernel catalog and repeated-measurement history tests."""

from __future__ import annotations

from pathlib import Path

from conftest import NOW, digest, seed_lineage

from atrex_runtime.domain.ids import (
    new_attempt_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
)
from atrex_runtime.domain.models import (
    Attempt,
    AttemptStatus,
    BranchRole,
    ChallengerProposalType,
    Epoch,
    EpochChallenger,
    EpochSelection,
    EpochStatus,
    KernelAgentRevision,
    KernelEvaluation,
    KernelMeasurement,
    KernelMeasurementPurpose,
    KernelRevision,
)
from atrex_runtime.registry.sqlite import SqliteRegistry


def test_catalog_joins_kernel_to_agent_attempt_and_durable_measurements(
    tmp_path: Path,
) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry, challenger_count=0)
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
            trajectories_per_branch=lineage.trajectories_per_branch,
            attempts_per_trajectory=lineage.attempts_per_trajectory,
            status=EpochStatus.RUNNING,
            winner_kernel_agent_revision_id=None,
            best_kernel_revision_id=None,
            created_at=NOW,
            completed_at=None,
        )
        registry.insert_epoch(epoch)
        attempt_id = new_attempt_id()
        registry.insert_attempt(
            Attempt(
                id=attempt_id,
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
        )
        candidate = KernelRevision(
            id=new_kernel_revision_id(),
            parent_id=seeded.baseline.id,
            artifact_digest=digest("candidate-kernel"),
            produced_by_attempt_id=attempt_id,
            evaluation=KernelEvaluation(True, 90.0, digest("candidate-gateway")),
            created_at="2026-08-14T00:01:00+00:00",
        )
        registry.register_kernel_revision(candidate)
        registry.complete_attempt(
            attempt_id,
            candidate.id,
            accepted_as_branch_best=True,
            failure_reason=None,
        )
        measurement = KernelMeasurement(
            id="measurement-1",
            kernel_revision_id=candidate.id,
            purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
            repeat=0,
            correct=True,
            latency_us=89.0,
            gateway_result_digest=digest("repeat-gateway"),
            agate_job_id="agate-job-1",
            created_at="2026-08-14T00:02:00+00:00",
        )
        assert registry.record_kernel_measurement(measurement) == measurement
        assert registry.record_kernel_measurement(measurement) == measurement

        catalog = registry.list_lineage_kernels(lineage.id)
        campaign_catalog = registry.list_campaign_kernels(lineage.campaign_id)

        assert campaign_catalog == catalog
        assert [entry.revision.id for entry in catalog] == [seeded.baseline.id, candidate.id]
        assert [entry.revision_number for entry in catalog] == [0, 1]
        assert catalog[0].parent_revision_number is None
        assert catalog[0].improvement_over_parent_percent is None
        assert catalog[1].parent_revision_number == 0
        assert catalog[1].improvement_over_parent_percent == 10.0
        assert catalog[0].kernel_agent_revision_id == seeded.active_revision_id
        assert catalog[0].kernel_agent_revision_number == 0
        assert catalog[0].attempt_id is None
        assert catalog[1].kernel_agent_revision_id == seeded.active_revision_id
        assert catalog[1].attempt_id == attempt_id
        assert catalog[1].epoch_id == epoch.id
        assert catalog[1].branch is BranchRole.ACTIVE
        assert catalog[1].accepted_as_branch_best
        assert registry.list_kernel_measurements(candidate.id) == [measurement]
        assert measurement.gateway_result_digest in registry.list_referenced_artifact_digests()


def test_agent_catalog_versions_bootstrap_and_attached_challenger(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry)
        lineage = registry.get_lineage(seeded.lineage_id)
        epoch = Epoch(
            id=new_epoch_id(),
            lineage_id=lineage.id,
            number=1,
            active_kernel_agent_revision_id=seeded.active_revision_id,
            challenger_kernel_agent_revision_ids=(),
            starting_kernel_revision_id=seeded.baseline.id,
            evidence_checkpoint=lineage.evidence_checkpoint,
            challenger_count=1,
            trajectories_per_branch=lineage.trajectories_per_branch,
            attempts_per_trajectory=lineage.attempts_per_trajectory,
            status=EpochStatus.BUILDING_CHALLENGER,
            winner_kernel_agent_revision_id=None,
            best_kernel_revision_id=None,
            created_at=NOW,
            completed_at=None,
        )
        registry.insert_epoch(epoch)
        challenger = registry.register_kernel_agent_revision(
            KernelAgentRevision(
                id=new_kernel_agent_revision_id(),
                parent_id=seeded.active_revision_id,
                creation_key=f"epoch:{epoch.id}:challenger",
                dsl=lineage.dsl,
                optimizer_digest=digest("challenger-agent"),
                created_by="evolver",
                created_at="2026-08-14T00:01:00+00:00",
                evolution_trace_digest=digest("challenger-trace"),
            )
        )
        registry.attach_challenger(
            EpochChallenger(
                epoch.id,
                1,
                challenger.id,
                ChallengerProposalType.EVOLVED,
                seeded.active_revision_id,
                digest("challenger-trace"),
            )
        )

        agents = registry.list_lineage_agent_revisions(lineage.id)
        assert registry.list_campaign_agent_revisions(lineage.campaign_id) == agents
        assert [entry.revision_number for entry in agents] == [0, 1]
        assert agents[0].parent_revision_number is None
        assert agents[0].disposition == "baseline"
        assert agents[0].active
        assert agents[1].parent_revision_number == 0
        assert agents[1].introduced_epoch_id == epoch.id
        assert agents[1].disposition == "challenger"
        assert not agents[1].active
        assert registry.find_kernel_agent_lineage(challenger.id).id == lineage.id

        registry.fail_epoch(epoch.id, "test failure")
        assert registry.list_lineage_agent_revisions(lineage.id)[1].disposition == "failed"


def _empty_epoch(registry: SqliteRegistry, seeded, *, number: int) -> Epoch:
    lineage = registry.get_lineage(seeded.lineage_id)
    epoch = Epoch(
        id=new_epoch_id(),
        lineage_id=lineage.id,
        number=number,
        active_kernel_agent_revision_id=lineage.active_kernel_agent_revision_id,
        challenger_kernel_agent_revision_ids=(),
        starting_kernel_revision_id=lineage.best_kernel_revision_id,
        evidence_checkpoint=lineage.evidence_checkpoint,
        challenger_count=1,
        trajectories_per_branch=lineage.trajectories_per_branch,
        attempts_per_trajectory=lineage.attempts_per_trajectory,
        status=EpochStatus.BUILDING_CHALLENGER,
        winner_kernel_agent_revision_id=None,
        best_kernel_revision_id=None,
        created_at=NOW,
        completed_at=None,
    )
    registry.insert_epoch(epoch)
    return epoch


def _finish_empty_epoch(
    registry: SqliteRegistry,
    epoch: Epoch,
    winner: str,
) -> None:
    registry.transition_epoch(epoch.id, EpochStatus.READY, EpochStatus.RUNNING)
    registry.transition_epoch(epoch.id, EpochStatus.RUNNING, EpochStatus.SELECTING)
    registry.complete_epoch(epoch.id, EpochSelection(winner, epoch.starting_kernel_revision_id))


def _add_historical_agent(registry: SqliteRegistry, seeded):
    first = _empty_epoch(registry, seeded, number=1)
    historical = registry.register_kernel_agent_revision(
        KernelAgentRevision(
            id=new_kernel_agent_revision_id(),
            parent_id=seeded.active_revision_id,
            creation_key="historical-agent",
            dsl=registry.get_lineage(seeded.lineage_id).dsl,
            optimizer_digest=digest("historical-agent"),
            created_by="evolver",
            created_at=NOW,
            evolution_trace_digest=digest("historical-trace"),
        )
    )
    registry.attach_challenger(
        EpochChallenger(
            first.id,
            1,
            historical.id,
            ChallengerProposalType.EVOLVED,
            seeded.active_revision_id,
            digest("historical-trace"),
        )
    )
    _finish_empty_epoch(registry, first, seeded.active_revision_id)
    checkpoint = registry.get_lineage(seeded.lineage_id).evidence_checkpoint
    registry.advance_lineage_evidence(
        seeded.lineage_id,
        checkpoint,
        digest("epoch-1-evidence"),
    )
    return historical


def test_reuse_historical_agent_competes_without_creating_a_new_version(
    tmp_path: Path,
) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry)
        historical = _add_historical_agent(registry, seeded)
        second = _empty_epoch(registry, seeded, number=2)
        registry.attach_challenger(
            EpochChallenger(
                second.id,
                1,
                historical.id,
                ChallengerProposalType.REUSE,
                historical.id,
                digest("reuse-trace"),
            )
        )

        assert len(registry.list_lineage_agent_revisions(seeded.lineage_id)) == 2
        assert registry.list_epoch_challengers(second.id)[0].proposal_type is (
            ChallengerProposalType.REUSE
        )
        _finish_empty_epoch(registry, second, historical.id)
        catalog = registry.list_lineage_agent_revisions(seeded.lineage_id)
        assert catalog[1].active is True
        assert catalog[1].disposition == "promoted"


def test_evolve_from_history_creates_a_child_of_the_selected_historical_revision(
    tmp_path: Path,
) -> None:
    with SqliteRegistry(tmp_path / "registry.sqlite") as registry:
        seeded = seed_lineage(registry)
        historical = _add_historical_agent(registry, seeded)
        second = _empty_epoch(registry, seeded, number=2)
        derived = registry.register_kernel_agent_revision(
            KernelAgentRevision(
                id=new_kernel_agent_revision_id(),
                parent_id=historical.id,
                creation_key="derived-from-history",
                dsl=historical.dsl,
                optimizer_digest=digest("derived-from-history"),
                created_by="evolver",
                created_at=NOW,
                evolution_trace_digest=digest("derived-trace"),
            )
        )
        registry.attach_challenger(
            EpochChallenger(
                second.id,
                1,
                derived.id,
                ChallengerProposalType.EVOLVE_FROM_HISTORY,
                historical.id,
                digest("derived-trace"),
            )
        )

        entries = registry.list_lineage_agent_revisions(seeded.lineage_id)
        assert [entry.revision_number for entry in entries] == [0, 1, 2]
        assert entries[2].revision.parent_id == historical.id
        assert entries[2].parent_revision_number == 1
