"""Pure Kernel Agent and Kernel selection policies."""

from __future__ import annotations

import asyncio
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from .domain.models import (
    AgentSelectionReason,
    BranchRole,
    BranchScore,
    KernelMeasurementPurpose,
    KernelRevision,
)
from .ports import (
    AttemptCandidateResult,
    KernelComparator,
    KernelComparisonResult,
    KernelMeasurementRun,
    KernelMeasurementRunner,
    KernelPairMeasurementRunner,
)


@dataclass(frozen=True, slots=True)
class AgentSelectionResult:
    """The winning Branch score and the rule that resolved the comparison."""

    winner: BranchScore
    reason: AgentSelectionReason


class OrdinaryEvaluateKernelComparator(KernelComparator):
    """Compare arithmetic means from concurrent ordinary Gateway evaluations."""

    def __init__(
        self,
        runner: KernelMeasurementRunner,
        *,
        repeats: int,
        measurement_uncertainty_us: float,
        purpose: KernelMeasurementPurpose,
    ) -> None:
        if repeats <= 0:
            raise ValueError("ordinary Evaluate comparison repeats must be positive")
        if measurement_uncertainty_us < 0:
            raise ValueError("measurement uncertainty cannot be negative")
        self._runner = runner
        self._repeats = repeats
        self._measurement_uncertainty_us = measurement_uncertainty_us
        self._purpose = purpose

    async def compare(
        self,
        incumbent: KernelRevision,
        candidate: KernelRevision,
    ) -> KernelComparisonResult:
        """Require complete correct measurement sets and compare their means."""

        async with asyncio.TaskGroup() as tasks:
            incumbent_tasks = tuple(
                tasks.create_task(self._runner.run(incumbent, repeat, self._purpose))
                for repeat in range(self._repeats)
            )
            candidate_tasks = tuple(
                tasks.create_task(self._runner.run(candidate, repeat, self._purpose))
                for repeat in range(self._repeats)
            )
        incumbent_runs = tuple(task.result() for task in incumbent_tasks)
        candidate_runs = tuple(task.result() for task in candidate_tasks)
        self._runner.aggregate(incumbent, incumbent_runs, self._purpose)
        candidate_result_digest = self._runner.aggregate(
            candidate,
            candidate_runs,
            self._purpose,
        )
        incumbent_mean = _ordinary_measurement_mean(
            incumbent_runs,
            self._repeats,
        )
        candidate_mean = _ordinary_measurement_mean(
            candidate_runs,
            self._repeats,
        )
        authoritative_candidate = (
            AttemptCandidateResult(
                artifact_digest=candidate.artifact_digest,
                gateway_result_digest=candidate_result_digest,
                correct=candidate_mean is not None,
                latency_us=candidate_mean,
            )
            if self._purpose is KernelMeasurementPurpose.KERNEL_RETENTION
            else None
        )
        if incumbent_mean is None or candidate_mean is None:
            return KernelComparisonResult(
                False,
                "not every repeated evaluate measurement passed",
                authoritative_candidate,
            )
        if incumbent_mean - candidate_mean <= self._measurement_uncertainty_us:
            return KernelComparisonResult(
                False,
                "candidate mean improvement did not exceed measurement uncertainty",
                authoritative_candidate,
            )
        return KernelComparisonResult(
            True,
            "candidate passed repeated evaluate comparison",
            authoritative_candidate,
        )


class SameAllocationAbbaKernelComparator(KernelComparator):
    """Compare geometric means from exact same-allocation interleaved runs."""

    def __init__(
        self,
        runner: KernelPairMeasurementRunner,
        *,
        repeats: int,
        minimum_improvement_percent: float,
        per_run_timeout_seconds: float,
        allocation_timeout_seconds: float,
        shape_batch_size: int,
        max_parallel_shape_batches: int,
        purpose: KernelMeasurementPurpose,
    ) -> None:
        if repeats <= 0:
            raise ValueError("same-allocation ABBA repeats must be positive")
        if not 0 <= minimum_improvement_percent < 100:
            raise ValueError("minimum improvement percent must be in [0, 100)")
        if per_run_timeout_seconds <= 0 or allocation_timeout_seconds <= 0:
            raise ValueError("same-allocation ABBA timeouts must be positive")
        if shape_batch_size <= 0 or max_parallel_shape_batches <= 0:
            raise ValueError("same-allocation ABBA batch limits must be positive")
        self._runner = runner
        self._repeats = repeats
        self._minimum_improvement_percent = minimum_improvement_percent
        self._per_run_timeout_seconds = per_run_timeout_seconds
        self._allocation_timeout_seconds = allocation_timeout_seconds
        self._shape_batch_size = shape_batch_size
        self._max_parallel_shape_batches = max_parallel_shape_batches
        self._purpose = purpose

    async def compare(
        self,
        incumbent: KernelRevision,
        candidate: KernelRevision,
    ) -> KernelComparisonResult:
        """Require every A/B run to pass, then apply a strict percentage gate."""
        result = await self._runner.run_pair(
            incumbent,
            candidate,
            repeats=self._repeats,
            purpose=self._purpose,
            per_run_timeout_seconds=self._per_run_timeout_seconds,
            allocation_timeout_seconds=self._allocation_timeout_seconds,
            shape_batch_size=self._shape_batch_size,
            max_parallel_shape_batches=self._max_parallel_shape_batches,
        )
        incumbent_mean = _measurement_geomean(result.incumbent_runs, self._repeats)
        candidate_mean = _measurement_geomean(result.candidate_runs, self._repeats)
        authoritative_candidate = (
            AttemptCandidateResult(
                artifact_digest=candidate.artifact_digest,
                gateway_result_digest=result.gateway_result_digest,
                correct=candidate_mean is not None,
                latency_us=candidate_mean,
            )
            if self._purpose is KernelMeasurementPurpose.KERNEL_RETENTION
            and result.gateway_result_digest is not None
            else None
        )
        if incumbent_mean is None or candidate_mean is None:
            return KernelComparisonResult(
                False,
                "not every authoritative same-allocation ABBA run passed",
                authoritative_candidate,
            )
        improvement = (incumbent_mean - candidate_mean) / incumbent_mean * 100.0
        if improvement <= self._minimum_improvement_percent:
            return KernelComparisonResult(
                False,
                f"candidate ABBA improvement {improvement:.6f}% did not exceed "
                f"{self._minimum_improvement_percent:.6f}%",
                authoritative_candidate,
            )
        return KernelComparisonResult(
            True,
            f"candidate passed same-allocation ABBA at {improvement:.6f}% improvement",
            authoritative_candidate,
        )


def _ordinary_measurement_mean(
    runs: tuple[KernelMeasurementRun, ...],
    repeats: int,
) -> float | None:
    """Validate exact repetitions and return their arithmetic mean."""
    if tuple(run.repeat for run in runs) != tuple(range(repeats)):
        return None
    values = [run.latency_us for run in runs if run.correct and run.latency_us is not None]
    if len(values) != repeats or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return statistics.fmean(values)


def _measurement_geomean(
    runs: tuple[KernelMeasurementRun, ...],
    repeats: int,
) -> float | None:
    """Validate exact correct repetitions and return their geometric mean."""
    if tuple(run.repeat for run in runs) != tuple(range(repeats)):
        return None
    values = [run.latency_us for run in runs if run.correct and run.latency_us is not None]
    if len(values) != repeats or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(statistics.fmean(math.log(value) for value in values))


class TrustedLatencyKernelComparator:
    """Compare authoritative Gateway aggregates outside the Agent sandbox.

    A deployment can replace this provider with paired same-allocation sampling
    without changing epoch control or an evolvable Kernel Agent Bundle.
    """

    def __init__(self, measurement_uncertainty_us: float) -> None:
        if measurement_uncertainty_us < 0:
            raise ValueError("measurement uncertainty cannot be negative")
        self._measurement_uncertainty_us = measurement_uncertainty_us

    async def compare(
        self,
        incumbent: KernelRevision,
        candidate: KernelRevision,
    ) -> KernelComparisonResult:
        """Require correctness and improvement beyond configured uncertainty."""
        incumbent_latency = incumbent.evaluation.latency_us
        candidate_latency = candidate.evaluation.latency_us
        if incumbent_latency is None:
            raise ValueError("incumbent Kernel has no comparable latency")
        if not candidate.evaluation.correct or candidate_latency is None:
            return KernelComparisonResult(False, "candidate failed correctness")
        if incumbent_latency - candidate_latency <= self._measurement_uncertainty_us:
            return KernelComparisonResult(
                False,
                "candidate improvement did not exceed measurement uncertainty",
            )
        return KernelComparisonResult(True, "candidate is a strict measured improvement")


def select_kernel_agent(
    incumbent: BranchScore,
    candidate: BranchScore,
    *,
    measurement_uncertainty_us: float = 0.0,
) -> AgentSelectionResult:
    """Select two Branch scores lexicographically, retaining the incumbent on a tie."""
    if candidate.branch is not BranchRole.CHALLENGER:
        raise ValueError("the candidate selection score must be a Challenger")
    if incumbent.kernel_agent_revision_id == candidate.kernel_agent_revision_id:
        raise ValueError("selection requires distinct Agent revisions")
    if measurement_uncertainty_us < 0:
        raise ValueError("measurement uncertainty cannot be negative")

    latency_delta = incumbent.best_latency_us - candidate.best_latency_us
    if abs(latency_delta) > measurement_uncertainty_us:
        return AgentSelectionResult(
            candidate if latency_delta > 0 else incumbent,
            AgentSelectionReason.LATENCY,
        )

    incumbent_secondary = (
        incumbent.first_best_attempt,
        -incumbent.strict_improvements,
        -incumbent.valid_candidates,
        incumbent.failed_candidates,
    )
    candidate_secondary = (
        candidate.first_best_attempt,
        -candidate.strict_improvements,
        -candidate.valid_candidates,
        candidate.failed_candidates,
    )
    if candidate_secondary < incumbent_secondary:
        return AgentSelectionResult(candidate, AgentSelectionReason.SECONDARY_CRITERIA)
    return AgentSelectionResult(incumbent, AgentSelectionReason.INCUMBENT_RETAINED)


def select_best_kernel(revisions: Iterable[KernelRevision]) -> KernelRevision:
    """Return the fastest correct Kernel, preserving input order on equal latency."""
    correct = [revision for revision in revisions if revision.evaluation.correct]
    if not correct:
        raise ValueError("Kernel selection requires at least one correct revision")
    return min(
        correct,
        key=lambda revision: (
            revision.evaluation.latency_us
            if revision.evaluation.latency_us is not None
            else float("inf")
        ),
    )
