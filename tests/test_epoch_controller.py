"""End-to-end tests for Active/Challenger epoch execution and recovery."""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

import anyio
import pytest
from conftest import NOW, FakeAttemptEvidence, digest, seed_lineage

from atrex_runtime.controller import EpochController
from atrex_runtime.domain.errors import InfrastructureError, InvalidTransitionError
from atrex_runtime.domain.ids import (
    AttemptId,
    KernelAgentRevisionId,
    new_kernel_revision_id,
    new_worker_session_id,
)
from atrex_runtime.domain.models import (
    AgentSelectionReason,
    AttemptReportStatus,
    AttemptStatus,
    BranchRole,
    CampaignStatus,
    EpochStatus,
    KernelEvaluation,
    KernelRevision,
    LineageStatus,
    TokenUsage,
    WorkerSession,
    WorkerSessionRole,
    WorkerSessionStatus,
)
from atrex_runtime.gateway.control import SqliteGatewayControl
from atrex_runtime.ports import (
    AttemptCandidateResult,
    BuildChallengerRequest,
    BuildChallengerResult,
    KernelAgentCandidate,
    KernelAgentCandidateProposal,
    KernelAgentReuseProposal,
    KernelComparisonResult,
    RunAttemptRequest,
    RunAttemptResult,
)
from atrex_runtime.registry.sqlite import SqliteRegistry


class FakeEvolver:
    """Deterministic Evolver that records idempotency keys."""

    def __init__(self) -> None:
        self.calls: list[BuildChallengerRequest] = []

    async def build_challenger(self, request: BuildChallengerRequest) -> BuildChallengerResult:
        self.calls.append(request)
        return BuildChallengerResult(
            KernelAgentCandidateProposal(
                "evolved",
                request.parent_revision.id,
                KernelAgentCandidate(
                    dsl=request.parent_revision.dsl,
                    optimizer_digest=digest("challenger-optimizer"),
                    runtime_state_digest=digest("challenger-runtime-state"),
                ),
            ),
            digest("evolution-trace"),
        )


@pytest.mark.anyio
@pytest.mark.parametrize("legacy_schema", [False, True])
async def test_managed_stop_preserves_completed_attempt_and_resumes_interrupted_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_schema: bool,
) -> None:
    database = tmp_path / "runtime.db"
    with monkeypatch.context() as patch:
        if legacy_schema:
            patch.setattr("atrex_runtime.registry.sqlite.migrate_stops", lambda _connection: None)
        with SqliteRegistry(database) as registry:
            seeded = seed_lineage(registry, challenger_count=0)
            controller = EpochController(
                registry, FakeEvolver(),
                ScriptedOptimizer(seeded.active_revision_id,
                                  active=[candidate("kept", 90), RuntimeError("killed")],
                                  challenger=[]),
                FakeAttemptEvidence(),
            )
            with pytest.raises(RuntimeError, match="killed"):
                await controller.run_epoch(seeded.lineage_id, 1)

    with SqliteRegistry(database) as registry:
        epoch = registry.find_epoch(seeded.lineage_id, 1)
        assert epoch is not None
        completed, running = registry.list_attempts(epoch.id)
        session_id = new_worker_session_id()
        registry.start_worker_session(WorkerSession(
            id=session_id, role=WorkerSessionRole.OPTIMIZER, subject_id=running.id,
            external_run_id="old-run", workspace_path=str(tmp_path),
            status=WorkerSessionStatus.RUNNING, started_at=NOW, attempt_id=running.id,
        ))
        registry.acquire_lineage_fence(
            seeded.lineage_id, "dead-scheduler", now=NOW,
            lease_expires_at="2999-01-01T00:00:00+00:00",
        )
        assert registry.stop_epoch(epoch.id, "operator stop").status is EpochStatus.STOPPED
        registry.stop_epoch(epoch.id, "operator stop")  # Idempotent.
        assert registry.get_attempt(completed.id) == completed
        interrupted = registry.get_attempt(running.id)
        assert interrupted.status is AttemptStatus.INTERRUPTED
        control = SqliteGatewayControl(tmp_path / "gateway.db", registry, signing_key=b"s" * 32)
        with pytest.raises(PermissionError, match="interrupted"):
            control.current_generation(running.id)
        control.close()
        assert interrupted.recovery_generation == running.recovery_generation
        assert interrupted.infrastructure_failures == running.infrastructure_failures
        assert registry.get_worker_session(session_id).status is WorkerSessionStatus.INTERRUPTED
        registry.acquire_lineage_fence(
            seeded.lineage_id, "new-scheduler", now=NOW,
            lease_expires_at="2999-01-01T00:00:00+00:00",
        )  # Old lease was invalidated even if its original expiration was in the future.
        optimizer = ScriptedOptimizer(seeded.active_revision_id,
                                      active=[candidate("resumed", 80)], challenger=[])
        result = await EpochController(
            registry, FakeEvolver(), optimizer, FakeAttemptEvidence(),
        ).run_epoch(seeded.lineage_id, 1)
        assert result.epoch.status is EpochStatus.COMPLETED
        assert optimizer.calls[BranchRole.ACTIVE] == [running.id]
        assert (
            registry.get_attempt(running.id).recovery_generation == running.recovery_generation + 1
        )
        assert registry.get_attempt(completed.id) == completed
        assert registry.stop_epoch(epoch.id, "stop completed").status is EpochStatus.COMPLETED
        assert registry.resume_stopped_epoch(epoch.id).status is EpochStatus.COMPLETED
    with closing(sqlite3.connect(database)) as connection:
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()


class NoChangeEvolver(FakeEvolver):
    """Invalid Evolver that returns the unchanged parent repository."""

    async def build_challenger(self, request: BuildChallengerRequest) -> BuildChallengerResult:
        build = await super().build_challenger(request)
        return BuildChallengerResult(
            KernelAgentCandidateProposal(
                "evolved",
                request.parent_revision.id,
                KernelAgentCandidate(
                    dsl=request.parent_revision.dsl,
                    optimizer_digest=request.parent_revision.optimizer_digest,
                ),
            ),
            build.evolution_trace_digest,
        )


class ReuseEvolver(FakeEvolver):
    """Select one existing historical Agent without creating a repository copy."""

    def __init__(self, revision_id: KernelAgentRevisionId) -> None:
        super().__init__()
        self._revision_id = revision_id

    async def build_challenger(self, request: BuildChallengerRequest) -> BuildChallengerResult:
        self.calls.append(request)
        return BuildChallengerResult(
            KernelAgentReuseProposal("reuse", self._revision_id),
            digest("reuse-evolution-trace"),
        )


Outcome = RunAttemptResult | BaseException


class ScriptedOptimizer:
    """Return branch-specific outcomes while recording durable Attempt identities."""

    def __init__(
        self,
        active_revision_id: KernelAgentRevisionId,
        *,
        active: Iterable[Outcome],
        challenger: Iterable[Outcome],
    ) -> None:
        self._active_revision_id = active_revision_id
        self._outcomes = {
            BranchRole.ACTIVE: deque(active),
            BranchRole.CHALLENGER: deque(challenger),
        }
        self.calls: dict[BranchRole, list[AttemptId]] = defaultdict(list)
        self.requests: dict[BranchRole, list[RunAttemptRequest]] = defaultdict(list)

    async def run_attempt(self, request: RunAttemptRequest) -> RunAttemptResult:
        branch = (
            BranchRole.ACTIVE
            if request.kernel_agent_revision_id == self._active_revision_id
            else BranchRole.CHALLENGER
        )
        self.calls[branch].append(request.attempt_id)
        self.requests[branch].append(request)
        outcome = self._outcomes[branch].popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FixedKernelComparator:
    """Return one trusted decision while recording each compared Kernel pair."""

    def __init__(
        self,
        accepted: bool,
        authoritative_candidate: AttemptCandidateResult | None = None,
    ) -> None:
        self._accepted = accepted
        self._authoritative_candidate = authoritative_candidate
        self.calls: list[tuple[KernelRevision, KernelRevision]] = []

    async def compare(
        self,
        incumbent: KernelRevision,
        candidate: KernelRevision,
    ) -> KernelComparisonResult:
        self.calls.append((incumbent, candidate))
        return KernelComparisonResult(
            self._accepted,
            "fixed test decision",
            self._authoritative_candidate,
        )


class FailingKernelComparator:
    """Fail after candidate registration to emulate interrupted Runtime finalization."""

    async def compare(
        self,
        _incumbent: KernelRevision,
        _candidate: KernelRevision,
    ) -> KernelComparisonResult:
        raise InfrastructureError("authoritative comparison unavailable")


class ReportAwareKernelComparator:
    """Require a durable candidate-ready Report before every retention call."""

    def __init__(self, registry: SqliteRegistry) -> None:
        self._registry = registry
        self.calls = 0

    async def compare(
        self,
        _incumbent: KernelRevision,
        candidate: KernelRevision,
    ) -> KernelComparisonResult:
        self.calls += 1
        assert candidate.produced_by_attempt_id is not None
        attempt = self._registry.get_attempt(candidate.produced_by_attempt_id)
        assert attempt.attempt_report_status is AttemptReportStatus.CANDIDATE_READY
        return KernelComparisonResult(True, "terminal Agent handoff is durable")


class ConcurrencyProbeOptimizer:
    """Require two Optimizer calls to overlap before either can complete."""

    def __init__(self) -> None:
        self._both_started = anyio.Event()
        self.running = 0
        self.peak_running = 0
        self.calls = 0

    async def run_attempt(self, _request: RunAttemptRequest) -> RunAttemptResult:
        self.calls += 1
        self.running += 1
        self.peak_running = max(self.peak_running, self.running)
        if self.calls == 2:
            self._both_started.set()
        with anyio.fail_after(1):
            await self._both_started.wait()
        self.running -= 1
        return candidate(f"concurrent-{self.calls}", 90 - self.calls)


class BranchConcurrencyProbeOptimizer:
    """Hold the first configured Branch wave until every slot is occupied."""

    def __init__(self, release_at: int) -> None:
        self._release_at = release_at
        self._release = anyio.Event()
        self.running = 0
        self.peak_running = 0
        self.calls = 0

    async def run_attempt(self, _request: RunAttemptRequest) -> RunAttemptResult:
        self.calls += 1
        call = self.calls
        self.running += 1
        self.peak_running = max(self.peak_running, self.running)
        if self.running == self._release_at:
            self._release.set()
        with anyio.fail_after(1):
            await self._release.wait()
        self.running -= 1
        return candidate(f"parallel-branch-{call}", 100 - call)


def candidate(label: str, latency_us: float) -> RunAttemptResult:
    """Build one correct evaluated candidate result."""
    return RunAttemptResult(
        candidate=AttemptCandidateResult(
            artifact_digest=digest(f"{label}-kernel"),
            gateway_result_digest=digest(f"{label}-gateway"),
            correct=True,
            latency_us=latency_us,
        ),
        attempt_report_digest=digest(f"{label}-attempt-report"),
        attempt_report_status=AttemptReportStatus.CANDIDATE_READY,
    )


@pytest.mark.anyio
async def test_first_epoch_same_agent_runs_two_branches_without_evolver(tmp_path: Path) -> None:
    with SqliteRegistry(tmp_path / "runtime.db") as registry:
        seeded = seed_lineage(
            registry,
            challenger_start_epoch=2,
            first_epoch_same_agent=True,
            attempts_per_trajectory=1,
        )
        evolver = FakeEvolver()
        optimizer = BranchConcurrencyProbeOptimizer(release_at=2)
        controller = EpochController(registry, evolver, optimizer, FakeAttemptEvidence())
        first = await controller.run_epoch(seeded.lineage_id, 1)
        assert optimizer.peak_running == 2
        assert evolver.calls == []
        attempts = registry.list_attempts(first.epoch.id)
        assert {attempt.branch for attempt in attempts} == {
            BranchRole.ACTIVE,
            BranchRole.CHALLENGER,
        }
        assert {attempt.kernel_agent_revision_id for attempt in attempts} == {
            seeded.active_revision_id
        }
        assert {attempt.input_kernel_revision_id for attempt in attempts} == {seeded.baseline.id}
        assert len(registry.list_lineage_agent_revisions(seeded.lineage_id)) == 1
        proposal = registry.list_epoch_challengers(first.epoch.id)[0]
        assert proposal.proposal_type.value == "replica"
        assert proposal.evolution_trace_digest is None
        assert first.epoch.winner_kernel_agent_revision_id == seeded.active_revision_id
        assert await controller.run_epoch(seeded.lineage_id, 1) == first
        assert optimizer.calls == 2
        lineage = registry.get_lineage(seeded.lineage_id)
        registry.advance_lineage_evidence(lineage.id, lineage.evidence_checkpoint, digest("epoch2"))
        second = await controller.run_epoch(seeded.lineage_id, 2)
        assert len(evolver.calls) == 1
        assert second.epoch.challenger_kernel_agent_revision_ids != (seeded.active_revision_id,)


@pytest.mark.anyio
async def test_epoch_without_challengers_runs_parallel_trajectories_from_same_kernel(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(
        registry,
        challenger_count=0,
        trajectories_per_branch=2,
        attempts_per_trajectory=2,
    )
    evolver = FakeEvolver()
    optimizer = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[
            candidate("trajectory-1-attempt-1", 90),
            candidate("trajectory-1-attempt-2", 80),
            candidate("trajectory-2-attempt-1", 95),
            candidate("trajectory-2-attempt-2", 70),
        ],
        challenger=[],
    )
    finished: list[tuple[int, int, int, AttemptStatus]] = []

    result = await EpochController(
        registry,
        evolver,
        optimizer,
        FakeAttemptEvidence(),
        attempt_finished=lambda _epoch, attempt: finished.append(
            (
                attempt.challenger_ordinal,
                attempt.trajectory_ordinal,
                attempt.ordinal,
                attempt.status,
            )
        ),
    ).run_epoch(seeded.lineage_id, 1)

    assert evolver.calls == []
    attempts = registry.list_attempts(result.epoch.id)
    assert len(attempts) == 4
    by_trajectory = {
        trajectory: sorted(
            (attempt for attempt in attempts if attempt.trajectory_ordinal == trajectory),
            key=lambda attempt: attempt.ordinal,
        )
        for trajectory in (1, 2)
    }
    assert all(
        items[0].input_kernel_revision_id == seeded.baseline.id for items in by_trajectory.values()
    )
    for items in by_trajectory.values():
        assert items[0].output_kernel_revision_id is not None
        assert items[1].input_kernel_revision_id == items[0].output_kernel_revision_id
    assert sorted(finished) == [
        (0, 1, 1, AttemptStatus.COMPLETED),
        (0, 1, 2, AttemptStatus.COMPLETED),
        (0, 2, 1, AttemptStatus.COMPLETED),
        (0, 2, 2, AttemptStatus.COMPLETED),
    ]
    assert result.epoch.winner_kernel_agent_revision_id == seeded.active_revision_id
    assert result.epoch.selection_reason is None
    registry.close()


@pytest.mark.anyio
async def test_event_only_lineage_runs_unevolved_on_its_own_kernel_line(
    tmp_path: Path,
) -> None:
    """The ablation arm must never evolve and must advance only on its own output."""
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(
        registry,
        challenger_count=0,
        trajectories_per_branch=1,
        attempts_per_trajectory=2,
        ephemeral_agent_state=True,
    )
    evolver = FakeEvolver()
    optimizer = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[candidate("attempt-1", 90), candidate("attempt-2", 80)],
        challenger=[],
    )

    result = await EpochController(
        registry,
        evolver,
        optimizer,
        FakeAttemptEvidence(),
    ).run_epoch(seeded.lineage_id, 1)

    assert evolver.calls == []
    assert result.epoch.challenger_kernel_agent_revision_ids == ()
    assert result.epoch.winner_kernel_agent_revision_id == seeded.active_revision_id
    attempts = registry.list_attempts(result.epoch.id)
    assert [attempt.branch for attempt in attempts] == [BranchRole.ACTIVE, BranchRole.ACTIVE]
    assert attempts[0].kernel_agent_revision_id == seeded.active_revision_id
    assert attempts[1].kernel_agent_revision_id == seeded.active_revision_id
    # The second Attempt continues from the first's accepted output, not the seed.
    assert attempts[1].input_kernel_revision_id == attempts[0].output_kernel_revision_id
    lineage = registry.get_lineage(seeded.lineage_id)
    assert lineage.ephemeral_agent_state is True
    assert lineage.active_kernel_agent_revision_id == seeded.active_revision_id
    assert lineage.best_kernel_revision_id == attempts[1].output_kernel_revision_id
    registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize("ephemeral_agent_state", [True, False])
async def test_pooled_event_only_lineage_pools_every_trajectory_into_one_baseline(
    tmp_path: Path,
    ephemeral_agent_state: bool,
) -> None:
    """Pool and Pool-Retained select the best Kernel across all Trajectories."""
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(
        registry,
        challenger_count=0,
        trajectories_per_branch=2,
        attempts_per_trajectory=2,
        ephemeral_agent_state=ephemeral_agent_state,
    )
    evolver = FakeEvolver()
    optimizer = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[
            candidate("trajectory-1-attempt-1", 95),
            candidate("trajectory-1-attempt-2", 90),
            candidate("trajectory-2-attempt-1", 85),
            # The overall best lands in the second Trajectory.
            candidate("trajectory-2-attempt-2", 60),
        ],
        challenger=[],
    )

    result = await EpochController(
        registry,
        evolver,
        optimizer,
        FakeAttemptEvidence(),
    ).run_epoch(seeded.lineage_id, 1)

    assert evolver.calls == []
    attempts = registry.list_attempts(result.epoch.id)
    assert len(attempts) == 4
    assert {attempt.trajectory_ordinal for attempt in attempts} == {1, 2}
    fastest = min(
        (
            registry.get_kernel_revision(attempt.output_kernel_revision_id)
            for attempt in attempts
            if attempt.output_kernel_revision_id is not None
        ),
        key=lambda kernel: kernel.evaluation.latency_us or float("inf"),
    )
    lineage = registry.get_lineage(seeded.lineage_id)
    # Both Trajectories feed one pooled baseline, unlike the isolated arm's own line.
    assert lineage.best_kernel_revision_id == fastest.id
    assert result.epoch.best_kernel_revision_id == fastest.id
    assert lineage.active_kernel_agent_revision_id == seeded.active_revision_id
    registry.close()


@pytest.mark.anyio
async def test_retention_comparison_finalizes_candidate_evaluation_before_completion(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(
        registry,
        challenger_count=0,
        trajectories_per_branch=1,
        attempts_per_trajectory=1,
    )
    abba_result = digest("abba-authority")
    await EpochController(
        registry,
        FakeEvolver(),
        ScriptedOptimizer(
            seeded.active_revision_id,
            active=[candidate("provisional", 90.0)],
            challenger=[],
        ),
        FakeAttemptEvidence(),
        kernel_retention_comparator=FixedKernelComparator(
            True,
            AttemptCandidateResult(
                digest("provisional-kernel"),
                abba_result,
                True,
                80.0,
            ),
        ),
    ).run_epoch(seeded.lineage_id, 1)

    attempt = registry.list_attempts(registry.list_epochs(seeded.lineage_id)[0].id)[0]
    assert attempt.output_kernel_revision_id is not None
    output = registry.get_kernel_revision(attempt.output_kernel_revision_id)
    assert output.evaluation == KernelEvaluation(True, 80.0, abba_result)
    assert attempt.accepted_as_branch_best is True
    registry.close()


@pytest.mark.anyio
async def test_trajectories_within_one_branch_execute_concurrently(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(
        registry,
        challenger_count=0,
        trajectories_per_branch=2,
        attempts_per_trajectory=1,
    )
    optimizer = ConcurrencyProbeOptimizer()

    await EpochController(
        registry,
        FakeEvolver(),
        optimizer,
        FakeAttemptEvidence(),
    ).run_epoch(seeded.lineage_id, 1)

    assert optimizer.calls == 2
    assert optimizer.peak_running == 2
    registry.close()


@pytest.mark.anyio
@pytest.mark.parametrize(("limit", "expected_peak"), [(1, 1), (2, 2), (3, 3)])
async def test_active_and_challenger_branches_run_with_configured_concurrency(
    tmp_path: Path,
    limit: int,
    expected_peak: int,
) -> None:
    registry = SqliteRegistry(tmp_path / f"runtime-{limit}.db")
    seeded = seed_lineage(
        registry,
        challenger_count=2,
        trajectories_per_branch=1,
        attempts_per_trajectory=1,
    )
    optimizer = BranchConcurrencyProbeOptimizer(expected_peak)

    await EpochController(
        registry,
        FakeEvolver(),
        optimizer,
        FakeAttemptEvidence(),
        max_parallel_branches=limit,
    ).run_epoch(seeded.lineage_id, 1)

    assert optimizer.calls == 3
    assert optimizer.peak_running == expected_peak
    registry.close()


@pytest.mark.anyio
async def test_multiple_challengers_are_built_sequentially_with_expanding_visibility(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(
        registry,
        challenger_count=2,
        attempts_per_trajectory=1,
    )
    evolver = FakeEvolver()
    result = await EpochController(
        registry,
        evolver,
        ScriptedOptimizer(
            seeded.active_revision_id,
            active=[candidate("active", 90)],
            challenger=[candidate("challenger-1", 80), candidate("challenger-2", 70)],
        ),
        FakeAttemptEvidence(),
    ).run_epoch(seeded.lineage_id, 1)

    assert len(evolver.calls) == 2
    assert [len(call.agent_catalog) for call in evolver.calls] == [1, 2]
    assert [len(call.kernel_catalog) for call in evolver.calls] == [1, 1]
    assert evolver.calls[0].agent_catalog[0].revision_number == 0
    assert evolver.calls[0].kernel_catalog[0].revision_number == 0
    assert evolver.calls[0].agent_catalog[0].revision.id == seeded.active_revision_id
    first_challenger = result.epoch.challenger_kernel_agent_revision_ids[0]
    assert {entry.revision.id for entry in evolver.calls[1].agent_catalog} == {
        seeded.active_revision_id,
        first_challenger,
    }
    assert len(result.epoch.challenger_kernel_agent_revision_ids) == 2
    assert result.epoch.winner_kernel_agent_revision_id in (
        result.epoch.challenger_kernel_agent_revision_ids
    )
    registry.close()


@pytest.mark.anyio
async def test_epoch_rejects_unchanged_evolver_candidate(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(registry, attempts_per_trajectory=1)
    controller = EpochController(
        registry,
        NoChangeEvolver(),
        ScriptedOptimizer(seeded.active_revision_id, active=[], challenger=[]),
        FakeAttemptEvidence(),
    )

    with pytest.raises(ValueError, match="incomplete Agent Bundle"):
        await controller.run_epoch(
            seeded.lineage_id,
            1,
        )
    failed = registry.find_epoch(seeded.lineage_id, 1)
    assert failed is not None
    assert failed.status is EpochStatus.FAILED

    recovery = registry.recover_failed_epoch(
        failed.id,
        recovery_key="replace-evolver",
        reason="fixed Evolver configuration",
    )
    assert recovery.attempt_ids == ()
    assert registry.get_epoch(failed.id).status is EpochStatus.BUILDING_CHALLENGER
    registry.close()


@pytest.mark.anyio
async def test_operator_recovery_resumes_failed_epoch_with_same_attempt_identity(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(registry, attempts_per_trajectory=1)
    evolver = FakeEvolver()
    failing = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[InfrastructureError("worker host lost")],
        challenger=[candidate("sibling-completed-before-recovery", 80)],
    )
    controller = EpochController(
        registry,
        evolver,
        failing,
        FakeAttemptEvidence(),
        max_infrastructure_retries=0,
    )

    with pytest.raises(InfrastructureError, match="exceeded infrastructure retry budget"):
        await controller.run_epoch(seeded.lineage_id, 1)

    failed_epoch = registry.find_epoch(seeded.lineage_id, 1)
    assert failed_epoch is not None
    failed_attempt = registry.find_attempt(
        failed_epoch.id,
        BranchRole.ACTIVE,
        0,
        1,
        1,
    )
    assert failed_attempt is not None
    sibling_attempt = registry.find_attempt(
        failed_epoch.id,
        BranchRole.CHALLENGER,
        1,
        1,
        1,
    )
    assert sibling_attempt is not None
    assert sibling_attempt.status is AttemptStatus.COMPLETED
    assert failed_epoch.status is EpochStatus.FAILED
    assert registry.get_lineage(seeded.lineage_id).status is LineageStatus.FAILED
    stale_fence = registry.acquire_lineage_fence(
        seeded.lineage_id,
        "crashed-scheduler",
        now="2026-08-14T00:00:00+00:00",
        lease_expires_at="2026-08-15T00:00:00+00:00",
    )

    recovery = registry.recover_failed_epoch(
        failed_epoch.id,
        recovery_key="incident-2026-08-14-001",
        reason="worker host was replaced",
    )
    assert (
        registry.recover_failed_epoch(
            failed_epoch.id,
            recovery_key="incident-2026-08-14-001",
            reason="worker host was replaced",
        )
        == recovery
    )
    recovered_attempt = registry.get_attempt(failed_attempt.id)
    assert recovery.attempt_ids == (failed_attempt.id,)
    assert recovery.generation == 1
    assert recovered_attempt.id == failed_attempt.id
    assert recovered_attempt.attempt_evidence_digest == failed_attempt.attempt_evidence_digest
    assert recovered_attempt.infrastructure_failures == 0
    assert recovered_attempt.recovery_generation == 1
    assert recovered_attempt.status.value == "running"
    assert registry.get_campaign(recovery.campaign_id).status is CampaignStatus.ACTIVE
    with pytest.raises(InvalidTransitionError, match="superseded"):
        registry.renew_lineage_fence(
            seeded.lineage_id,
            stale_fence,
            "crashed-scheduler",
            lease_expires_at="2026-08-16T00:00:00+00:00",
        )
    with pytest.raises(InvalidTransitionError, match="different reason"):
        registry.recover_failed_epoch(
            failed_epoch.id,
            recovery_key="incident-2026-08-14-001",
            reason="different justification",
        )

    resumed = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[candidate("recovered-active", 90)],
        challenger=[candidate("recovered-challenger", 80)],
    )
    result = await EpochController(
        registry,
        evolver,
        resumed,
        FakeAttemptEvidence(),
        max_infrastructure_retries=0,
    ).run_epoch(seeded.lineage_id, 1)

    assert result.epoch.status is EpochStatus.COMPLETED
    assert resumed.calls[BranchRole.ACTIVE] == [failed_attempt.id]
    assert registry.get_attempt(failed_attempt.id).recovery_generation == 1
    registry.close()
    with closing(sqlite3.connect(tmp_path / "runtime.db")) as connection:
        events = connection.execute(
            """SELECT kind FROM runtime_events
               WHERE kind IN ('attempt.recovered', 'epoch.recovered',
                              'lineage.recovered', 'campaign.reopened')
               ORDER BY sequence"""
        ).fetchall()
    assert events == [
        ("attempt.recovered",),
        ("epoch.recovered",),
        ("lineage.recovered",),
        ("campaign.reopened",),
    ]


@pytest.mark.anyio
async def test_operator_recovery_resumes_registered_candidate_finalization(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(
        registry,
        challenger_count=0,
        attempts_per_trajectory=1,
    )
    optimizer = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[candidate("registered-before-comparison-failure", 80)],
        challenger=[],
    )

    with pytest.raises(InfrastructureError, match="authoritative comparison unavailable"):
        await EpochController(
            registry,
            FakeEvolver(),
            optimizer,
            FakeAttemptEvidence(),
            kernel_retention_comparator=FailingKernelComparator(),
        ).run_epoch(seeded.lineage_id, 1)

    failed_epoch = registry.find_epoch(seeded.lineage_id, 1)
    assert failed_epoch is not None
    assert failed_epoch.status is EpochStatus.FAILED
    failed_attempt = registry.find_attempt(
        failed_epoch.id,
        BranchRole.ACTIVE,
        0,
        1,
        1,
    )
    assert failed_attempt is not None
    assert failed_attempt.status is AttemptStatus.RUNNING
    registered = registry.find_kernel_revision_by_attempt(failed_attempt.id)
    assert registered is not None
    assert failed_attempt.output_kernel_revision_id is None

    recovery = registry.recover_failed_epoch(
        failed_epoch.id,
        recovery_key="resume-authoritative-comparison",
        reason="comparison infrastructure recovered",
    )
    assert recovery.attempt_ids == (failed_attempt.id,)
    assert registry.get_epoch(failed_epoch.id).status is EpochStatus.RUNNING
    assert registry.get_attempt(failed_attempt.id).recovery_generation == 1

    unused_optimizer = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[],
        challenger=[],
    )
    result = await EpochController(
        registry,
        FakeEvolver(),
        unused_optimizer,
        FakeAttemptEvidence(),
        kernel_retention_comparator=FixedKernelComparator(True),
    ).run_epoch(seeded.lineage_id, 1)

    completed_attempt = registry.get_attempt(failed_attempt.id)
    assert result.epoch.status is EpochStatus.COMPLETED
    assert completed_attempt.output_kernel_revision_id == registered.id
    assert completed_attempt.accepted_as_branch_best is True
    assert unused_optimizer.calls == {}
    registry.close()


@pytest.mark.anyio
async def test_epoch_promotes_challenger_and_best_kernel_after_scoped_retry(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(
        registry,
        optimizer_model="optimizer-model",
        evolver_model="evolver-model",
    )
    evolver = FakeEvolver()
    optimizer = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[
            InfrastructureError("gateway unavailable"),
            candidate("a1", 90),
            candidate("a2", 85),
        ],
        challenger=[candidate("c1", 80), candidate("c2", 70)],
    )
    controller = EpochController(
        registry,
        evolver,
        optimizer,
        FakeAttemptEvidence(),
        max_infrastructure_retries=2,
    )

    result = await controller.run_epoch(
        seeded.lineage_id,
        1,
    )

    assert result.epoch.status is EpochStatus.COMPLETED
    assert (
        result.epoch.winner_kernel_agent_revision_id
        == result.challenger_scores[0].kernel_agent_revision_id
    )
    assert result.epoch.selection_reason is AgentSelectionReason.LATENCY
    assert result.epoch.best_kernel_revision_id is not None
    best = registry.get_kernel_revision(result.epoch.best_kernel_revision_id)
    assert best.evaluation.latency_us == 70
    lineage = registry.get_lineage(seeded.lineage_id)
    assert (
        lineage.active_kernel_agent_revision_id
        == result.challenger_scores[0].kernel_agent_revision_id
    )
    assert lineage.best_kernel_revision_id == best.id
    assert lineage.next_epoch_number == 2
    assert lineage.status is LineageStatus.AWAITING_EVIDENCE
    assert len(evolver.calls) == 1
    assert evolver.calls[0].model == "evolver-model"
    challenger = registry.get_kernel_agent_revision(
        result.challenger_scores[0].kernel_agent_revision_id
    )
    assert challenger.evolution_trace_digest == digest("evolution-trace")
    assert challenger.runtime_state_digest == digest("challenger-runtime-state")
    assert optimizer.calls[BranchRole.ACTIVE][0] == optimizer.calls[BranchRole.ACTIVE][1]
    assert (
        optimizer.requests[BranchRole.ACTIVE][0].attempt_evidence_digest
        == optimizer.requests[BranchRole.ACTIVE][1].attempt_evidence_digest
    )
    assert len(set(optimizer.calls[BranchRole.ACTIVE])) == 2
    attempts = registry.list_attempts(result.epoch.id)
    assert len(attempts) == 4
    for requests in optimizer.requests.values():
        for request in requests:
            assert request.model == "optimizer-model"
            persisted = registry.get_attempt(request.attempt_id)
            assert request.attempt_evidence_digest == persisted.attempt_evidence_digest
            assert request.epoch_evidence_checkpoint == result.epoch.evidence_checkpoint
    assert sum(attempt.infrastructure_failures for attempt in attempts) == 1
    retried = next(attempt for attempt in attempts if attempt.infrastructure_failures == 1)
    assert retried.recovery_generation == 1
    first_trace = registry.record_attempt_session_trace(
        attempts[0].id,
        digest("optimizer-session-one"),
        "completed",
        1000,
        TokenUsage(100, 20, 30, 10),
    )
    second_trace = registry.record_attempt_session_trace(
        attempts[0].id,
        digest("optimizer-session-two"),
        "process-exit-1",
        1000,
        TokenUsage(200, 40, 0, 0),
    )
    assert first_trace.run_ordinal == 1
    assert second_trace.run_ordinal == 2
    assert first_trace.token_usage.total_tokens == 160
    assert registry.list_attempt_session_traces(attempts[0].id) == [
        first_trace,
        second_trace,
    ]
    registry.close()


@pytest.mark.anyio
async def test_kernel_retention_policy_controls_branch_and_epoch_best(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(registry, attempts_per_trajectory=1)
    retention = FixedKernelComparator(False)
    result = await EpochController(
        registry,
        FakeEvolver(),
        ScriptedOptimizer(
            seeded.active_revision_id,
            active=[candidate("active-faster-but-rejected", 80)],
            challenger=[candidate("challenger-faster-but-rejected", 70)],
        ),
        FakeAttemptEvidence(),
        kernel_retention_comparator=retention,
    ).run_epoch(seeded.lineage_id, 1)

    assert len(retention.calls) == 2
    assert all(
        not attempt.accepted_as_branch_best for attempt in registry.list_attempts(result.epoch.id)
    )
    assert result.epoch.best_kernel_revision_id == seeded.baseline.id
    assert result.epoch.winner_kernel_agent_revision_id == seeded.active_revision_id
    registry.close()


@pytest.mark.anyio
async def test_agent_promotion_policy_is_independent_from_best_kernel(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(registry, attempts_per_trajectory=1)
    promotion = FixedKernelComparator(False)
    result = await EpochController(
        registry,
        FakeEvolver(),
        ScriptedOptimizer(
            seeded.active_revision_id,
            active=[candidate("active-retained", 90)],
            challenger=[candidate("challenger-retained", 70)],
        ),
        FakeAttemptEvidence(),
        agent_promotion_comparator=promotion,
    ).run_epoch(seeded.lineage_id, 1)

    assert len(promotion.calls) == 1
    active_kernel, challenger_kernel = promotion.calls[0]
    assert active_kernel.evaluation.latency_us == 90
    assert challenger_kernel.evaluation.latency_us == 70
    assert result.epoch.winner_kernel_agent_revision_id == seeded.active_revision_id
    assert result.epoch.best_kernel_revision_id == challenger_kernel.id
    lineage = registry.get_lineage(seeded.lineage_id)
    assert lineage.active_kernel_agent_revision_id == seeded.active_revision_id
    assert lineage.best_kernel_revision_id == challenger_kernel.id
    registry.close()


@pytest.mark.anyio
async def test_epoch_records_kernel_and_agent_rollbacks(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(registry)
    controller = EpochController(
        registry,
        FakeEvolver(),
        ScriptedOptimizer(
            seeded.active_revision_id,
            active=[candidate("active-best", 70), candidate("active-regression", 80)],
            challenger=[
                candidate("challenger-best", 90),
                candidate("challenger-regression", 95),
            ],
        ),
        FakeAttemptEvidence(),
    )

    result = await controller.run_epoch(seeded.lineage_id, 1)

    assert result.epoch.winner_kernel_agent_revision_id == seeded.active_revision_id
    events = registry.list_runtime_events(after_sequence=0, limit=200)
    rollbacks = [event for event in events if event.kind.endswith(".rollback")]
    assert [event.kind for event in rollbacks] == [
        "kernel.rollback",
        "kernel.rollback",
        "kernel_agent.rollback",
    ]
    assert all(event.payload["schema_version"] == 1 for event in rollbacks)
    registry.close()


@pytest.mark.anyio
async def test_running_attempt_resumes_without_repeating_registered_kernel(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(registry)
    evolver = FakeEvolver()
    crashing = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[RuntimeError("simulated process crash")],
        challenger=[],
    )
    first_controller = EpochController(registry, evolver, crashing, FakeAttemptEvidence())

    with pytest.raises(RuntimeError, match="simulated process crash"):
        await first_controller.run_epoch(
            seeded.lineage_id,
            1,
        )

    epoch = registry.find_epoch(seeded.lineage_id, 1)
    assert epoch is not None
    interrupted = registry.find_attempt(epoch.id, BranchRole.ACTIVE, 0, 1, 1)
    assert interrupted is not None
    registry.register_kernel_revision(
        KernelRevision(
            id=new_kernel_revision_id(),
            parent_id=interrupted.input_kernel_revision_id,
            artifact_digest=digest("recovered-output-kernel"),
            produced_by_attempt_id=interrupted.id,
            evaluation=KernelEvaluation(
                correct=True,
                latency_us=95,
                gateway_result_digest=digest("recovered-output-gateway"),
            ),
            created_at=NOW,
        )
    )

    registry.stop_epoch(epoch.id, "stop before pending Kernel acceptance")
    assert registry.get_attempt(interrupted.id).status is AttemptStatus.INTERRUPTED
    resumed_optimizer = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[candidate("active-second", 90)],
        challenger=[candidate("challenger-first", 80), candidate("challenger-second", 75)],
    )
    retention = ReportAwareKernelComparator(registry)
    resumed_controller = EpochController(
        registry,
        evolver,
        resumed_optimizer,
        FakeAttemptEvidence(),
        kernel_retention_comparator=retention,
    )
    result = await resumed_controller.run_epoch(
        seeded.lineage_id,
        1,
    )

    assert result.epoch.status is EpochStatus.COMPLETED
    assert len(evolver.calls) == 1
    assert interrupted.id not in resumed_optimizer.calls[BranchRole.ACTIVE]
    recovered = registry.get_attempt(interrupted.id)
    assert recovered.output_kernel_revision_id is not None
    assert not recovered.accepted_as_branch_best
    assert recovered.failure_reason == "candidate lacks a candidate_ready Attempt report"
    assert retention.calls == 3
    registry.close()


@pytest.mark.anyio
async def test_controller_reuses_historical_agent_without_allocating_a_version(
    tmp_path: Path,
) -> None:
    registry = SqliteRegistry(tmp_path / "runtime.db")
    seeded = seed_lineage(registry, attempts_per_trajectory=1)
    first_optimizer = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[candidate("epoch-1-active", 80)],
        challenger=[candidate("epoch-1-challenger", 100)],
    )
    first = await EpochController(
        registry,
        FakeEvolver(),
        first_optimizer,
        FakeAttemptEvidence(),
    ).run_epoch(seeded.lineage_id, 1)
    assert first.epoch.winner_kernel_agent_revision_id == seeded.active_revision_id
    historical = registry.list_lineage_agent_revisions(seeded.lineage_id)[1].revision
    checkpoint = registry.get_lineage(seeded.lineage_id).evidence_checkpoint
    registry.advance_lineage_evidence(
        seeded.lineage_id,
        checkpoint,
        digest("epoch-1-complete-evidence"),
    )

    second_optimizer = ScriptedOptimizer(
        seeded.active_revision_id,
        active=[candidate("epoch-2-active", 100)],
        challenger=[candidate("epoch-2-reused", 70)],
    )
    second = await EpochController(
        registry,
        ReuseEvolver(historical.id),
        second_optimizer,
        FakeAttemptEvidence(),
    ).run_epoch(seeded.lineage_id, 2)

    assert second.epoch.winner_kernel_agent_revision_id == historical.id
    assert len(registry.list_lineage_agent_revisions(seeded.lineage_id)) == 2
    proposal = registry.list_epoch_challengers(second.epoch.id)[0]
    assert proposal.proposal_type.value == "reuse"
    assert proposal.base_revision_id == historical.id
    registry.close()
