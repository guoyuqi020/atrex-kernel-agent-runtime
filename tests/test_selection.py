"""Pure policy tests for Kernel Agent selection."""

from __future__ import annotations

import asyncio
import math

import pytest
from conftest import NOW, digest

from atrex_runtime.domain.ids import new_kernel_agent_revision_id, new_kernel_revision_id
from atrex_runtime.domain.models import (
    AgentSelectionReason,
    BranchRole,
    BranchScore,
    KernelEvaluation,
    KernelMeasurementPurpose,
    KernelRevision,
)
from atrex_runtime.ports import KernelMeasurementRun, KernelPairMeasurementResult
from atrex_runtime.selection import (
    OrdinaryEvaluateKernelComparator,
    SameAllocationAbbaKernelComparator,
    TrustedLatencyKernelComparator,
    select_kernel_agent,
)


def score(branch: BranchRole, latency: float) -> BranchScore:
    """Construct a minimal branch score."""
    return BranchScore(
        branch=branch,
        challenger_ordinal=0 if branch is BranchRole.ACTIVE else 1,
        kernel_agent_revision_id=new_kernel_agent_revision_id(),
        best_latency_us=latency,
        first_best_attempt=1,
        strict_improvements=1,
        valid_candidates=1,
        failed_candidates=0,
    )


def test_exact_tie_retains_active() -> None:
    active = score(BranchRole.ACTIVE, 80)
    challenger = score(BranchRole.CHALLENGER, 80)

    selection = select_kernel_agent(active, challenger)

    assert selection.winner is active
    assert selection.reason is AgentSelectionReason.INCUMBENT_RETAINED


def test_measurement_uncertainty_uses_secondary_metrics() -> None:
    active = score(BranchRole.ACTIVE, 80)
    challenger = BranchScore(
        branch=BranchRole.CHALLENGER,
        challenger_ordinal=1,
        kernel_agent_revision_id=new_kernel_agent_revision_id(),
        best_latency_us=79.95,
        first_best_attempt=1,
        strict_improvements=2,
        valid_candidates=2,
        failed_candidates=0,
    )

    selection = select_kernel_agent(active, challenger, measurement_uncertainty_us=0.1)

    assert selection.winner is challenger
    assert selection.reason is AgentSelectionReason.SECONDARY_CRITERIA


def test_latency_beyond_uncertainty_reports_a_latency_decision() -> None:
    active = score(BranchRole.ACTIVE, 80)
    challenger = score(BranchRole.CHALLENGER, 70)

    selection = select_kernel_agent(active, challenger, measurement_uncertainty_us=0.1)

    assert selection.winner is challenger
    assert selection.reason is AgentSelectionReason.LATENCY

    slower = score(BranchRole.CHALLENGER, 90)
    retained = select_kernel_agent(active, slower, measurement_uncertainty_us=0.1)

    assert retained.winner is active
    assert retained.reason is AgentSelectionReason.LATENCY


@pytest.mark.anyio
async def test_trusted_comparator_requires_improvement_beyond_uncertainty() -> None:
    def kernel(label: str, latency: float) -> KernelRevision:
        return KernelRevision(
            id=new_kernel_revision_id(),
            parent_id=None,
            artifact_digest=digest(f"{label}-kernel"),
            produced_by_attempt_id=None,
            evaluation=KernelEvaluation(True, latency, digest(f"{label}-gateway")),
            created_at=NOW,
        )

    comparator = TrustedLatencyKernelComparator(0.1)

    assert not (await comparator.compare(kernel("base", 80), kernel("noise", 79.95))).accepted
    assert (await comparator.compare(kernel("base-2", 80), kernel("faster", 79.8))).accepted


@pytest.mark.anyio
async def test_repeated_evaluate_runs_concurrently_and_compares_arithmetic_means() -> None:
    def kernel(label: str, latency: float) -> KernelRevision:
        return KernelRevision(
            id=new_kernel_revision_id(),
            parent_id=None,
            artifact_digest=digest(f"{label}-kernel"),
            produced_by_attempt_id=None,
            evaluation=KernelEvaluation(True, latency, digest(f"{label}-gateway")),
            created_at=NOW,
        )

    incumbent = kernel("mean-base", 999)
    candidate = kernel("mean-candidate", 999)
    started: list[tuple[KernelRevision, int]] = []
    aggregates: list[tuple[KernelRevision, tuple[KernelMeasurementRun, ...]]] = []
    all_started = asyncio.Event()

    class Runner:
        async def run(
            self,
            revision: KernelRevision,
            repeat: int,
            purpose: KernelMeasurementPurpose,
        ) -> KernelMeasurementRun:
            assert purpose is KernelMeasurementPurpose.KERNEL_RETENTION
            started.append((revision, repeat))
            if len(started) == 6:
                all_started.set()
            await all_started.wait()
            latencies = (100.0, 102.0, 101.0) if revision is incumbent else (90.0, 93.0, 96.0)
            return KernelMeasurementRun(
                repeat,
                True,
                latencies[repeat],
            )

        def aggregate(
            self,
            revision: KernelRevision,
            runs: tuple[KernelMeasurementRun, ...],
            purpose: KernelMeasurementPurpose,
        ) -> object:
            assert purpose is KernelMeasurementPurpose.KERNEL_RETENTION
            aggregates.append((revision, runs))
            return digest(f"aggregate-{revision.id}")

    result = await asyncio.wait_for(
        OrdinaryEvaluateKernelComparator(
            Runner(),
            repeats=3,
            measurement_uncertainty_us=1.0,
            purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
        ).compare(incumbent, candidate),
        timeout=1,
    )

    assert len(started) == 6
    assert len(aggregates) == 2
    assert result.accepted
    authoritative = result.authoritative_candidate
    assert authoritative is not None
    assert authoritative.gateway_result_digest == digest(f"aggregate-{candidate.id}")
    assert authoritative.latency_us == 93.0


@pytest.mark.anyio
async def test_single_repeat_evaluate_still_measures_both_sides_and_finalizes_b() -> None:
    def kernel(label: str) -> KernelRevision:
        return KernelRevision(
            id=new_kernel_revision_id(),
            parent_id=None,
            artifact_digest=digest(f"{label}-kernel"),
            produced_by_attempt_id=None,
            evaluation=KernelEvaluation(True, 999.0, digest(f"{label}-old-result")),
            created_at=NOW,
        )

    incumbent = kernel("single-a")
    candidate = kernel("single-b")
    calls: list[KernelRevision] = []

    class Runner:
        async def run(
            self,
            revision: KernelRevision,
            repeat: int,
            purpose: KernelMeasurementPurpose,
        ) -> KernelMeasurementRun:
            assert repeat == 0
            assert purpose is KernelMeasurementPurpose.KERNEL_RETENTION
            calls.append(revision)
            return KernelMeasurementRun(0, True, 100.0 if revision is incumbent else 90.0)

        def aggregate(
            self,
            revision: KernelRevision,
            runs: tuple[KernelMeasurementRun, ...],
            purpose: KernelMeasurementPurpose,
        ) -> object:
            assert len(runs) == 1
            assert purpose is KernelMeasurementPurpose.KERNEL_RETENTION
            return digest(f"single-aggregate-{revision.id}")

    result = await OrdinaryEvaluateKernelComparator(
        Runner(),  # type: ignore[arg-type]
        repeats=1,
        measurement_uncertainty_us=0.0,
        purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
    ).compare(incumbent, candidate)

    assert set(calls) == {incumbent, candidate}
    assert result.accepted is True
    authoritative = result.authoritative_candidate
    assert authoritative is not None
    assert authoritative.latency_us == 90.0
    assert authoritative.gateway_result_digest == digest(f"single-aggregate-{candidate.id}")


@pytest.mark.anyio
async def test_same_allocation_abba_requires_all_runs_and_strict_percent_gain() -> None:
    def kernel(label: str) -> KernelRevision:
        return KernelRevision(
            id=new_kernel_revision_id(),
            parent_id=None,
            artifact_digest=digest(f"{label}-kernel"),
            produced_by_attempt_id=None,
            evaluation=KernelEvaluation(True, 999, digest(f"{label}-gateway")),
            created_at=NOW,
        )

    incumbent = kernel("abba-base")
    candidate = kernel("abba-candidate")

    class Runner:
        async def run_pair(self, *args: object, **kwargs: object) -> KernelPairMeasurementResult:
            assert args == (incumbent, candidate)
            assert kwargs["repeats"] == 2
            assert kwargs["purpose"] is KernelMeasurementPurpose.KERNEL_RETENTION
            return KernelPairMeasurementResult(
                (
                    KernelMeasurementRun(0, True, 100.0),
                    KernelMeasurementRun(1, True, 102.0),
                ),
                (
                    KernelMeasurementRun(0, True, 90.0),
                    KernelMeasurementRun(1, True, 92.0),
                ),
                gateway_result_digest=digest("abba-result"),
            )

    comparator = SameAllocationAbbaKernelComparator(
        Runner(),  # type: ignore[arg-type]
        repeats=2,
        minimum_improvement_percent=5.0,
        per_run_timeout_seconds=100,
        allocation_timeout_seconds=500,
        shape_batch_size=4,
        max_parallel_shape_batches=2,
        purpose=KernelMeasurementPurpose.KERNEL_RETENTION,
    )

    result = await comparator.compare(incumbent, candidate)

    assert result.accepted
    assert "ABBA" in result.reason
    authoritative = result.authoritative_candidate
    assert authoritative is not None
    assert authoritative.artifact_digest == candidate.artifact_digest
    assert authoritative.gateway_result_digest == digest("abba-result")
    assert authoritative.correct is True
    assert authoritative.latency_us == pytest.approx(math.sqrt(90.0 * 92.0))
