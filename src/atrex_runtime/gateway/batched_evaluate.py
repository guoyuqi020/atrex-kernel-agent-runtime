"""Shared trusted Shape-batched execution for ordinary Agate Evaluate jobs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import anyio

from ..artifacts.local import JsonValue
from .contract import AgateEvaluationContractV1
from .correctness import merge_correctness_summaries
from .private_results import project_candidate_rejection
from .protocol import EvaluationV2

EVALUATE_SHAPE_BATCH_SIZE = 4
EVALUATE_MAX_PARALLEL_BATCHES = 4
CANDIDATE_REJECTED_CATEGORY = "candidate_rejected"


@dataclass(frozen=True, slots=True)
class ShapeBatch:
    """One deterministic subset of a sealed evaluation contract."""

    index: int
    shape_ids: tuple[str, ...]
    contract: AgateEvaluationContractV1
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ShapeBatchOutcome:
    """Trusted result of one physical Agate Eval job."""

    job: JsonValue
    evaluation: EvaluationV2
    job_id: str | None = None
    worker_result: JsonValue | None = None


@dataclass(frozen=True, slots=True)
class BatchedEvaluateOutcome:
    """One logical Evaluate reconstructed from all physical Shape batches."""

    job: JsonValue
    evaluation: EvaluationV2
    job_id: str | None
    worker_result: JsonValue
    batches: tuple[ShapeBatchOutcome, ...]


class ShapeBatchedEvaluateExecutor:
    """Execute four Shapes per job with at most four concurrent Agate jobs."""

    def __init__(
        self,
        *,
        shape_batch_size: int = EVALUATE_SHAPE_BATCH_SIZE,
        max_parallel_batches: int = EVALUATE_MAX_PARALLEL_BATCHES,
    ) -> None:
        if shape_batch_size <= 0 or max_parallel_batches <= 0:
            raise ValueError("Evaluate Shape batch limits must be positive")
        self._shape_batch_size = shape_batch_size
        self._max_parallel_batches = max_parallel_batches

    async def run(
        self,
        contract: AgateEvaluationContractV1,
        idempotency_key: str,
        evaluate: Callable[[ShapeBatch], Awaitable[ShapeBatchOutcome]],
    ) -> BatchedEvaluateOutcome:
        """Run every Shape exactly once and aggregate correctness and latency."""
        batches = self._batches(contract, idempotency_key)
        results: list[ShapeBatchOutcome | None] = [None] * len(batches)
        limiter = anyio.Semaphore(self._max_parallel_batches)
        rejection: tuple[ShapeBatch, ShapeBatchOutcome] | None = None

        async with anyio.create_task_group() as tasks:

            async def run_one(batch: ShapeBatch) -> None:
                nonlocal rejection
                async with limiter:
                    outcome = await evaluate(batch)
                    results[batch.index] = outcome
                    if _candidate_rejection_detail(outcome) is not None:
                        if rejection is None or batch.index < rejection[0].index:
                            rejection = (batch, outcome)
                        tasks.cancel_scope.cancel()

            for batch in batches:
                tasks.start_soon(run_one, batch)
        if rejection is not None:
            return _candidate_rejected_outcome(
                rejection[0],
                rejection[1],
                batch_count=len(batches),
            )
        completed = tuple(result for result in results if result is not None)
        if len(completed) != len(batches):
            raise AssertionError("Shape-batched Evaluate produced no result")
        return self._aggregate(batches, completed)

    def _batches(
        self,
        contract: AgateEvaluationContractV1,
        idempotency_key: str,
    ) -> tuple[ShapeBatch, ...]:
        shape_ids = sorted_shape_ids(contract)
        groups = tuple(
            shape_ids[offset : offset + self._shape_batch_size]
            for offset in range(0, len(shape_ids), self._shape_batch_size)
        )
        single = len(groups) == 1
        return tuple(
            ShapeBatch(
                index=index,
                shape_ids=group,
                contract=subset_evaluation_contract(contract, group),
                idempotency_key=(
                    idempotency_key
                    if single
                    else _batch_idempotency_key(idempotency_key, index, group)
                ),
            )
            for index, group in enumerate(groups)
        )

    def _aggregate(
        self,
        batches: tuple[ShapeBatch, ...],
        outcomes: tuple[ShapeBatchOutcome, ...],
    ) -> BatchedEvaluateOutcome:
        if len(outcomes) == 1:
            outcome = outcomes[0]
            worker = outcome.worker_result or _worker_result(
                outcome.evaluation,
                batches,
                outcomes,
            )
            return BatchedEvaluateOutcome(
                outcome.job,
                outcome.evaluation,
                outcome.job_id,
                worker,
                outcomes,
            )

        evaluation = _aggregate_batch_evaluations(batches, outcomes)
        job = cast(
            JsonValue,
            {
                "schema_version": 1,
                "operation": "shape_batched_evaluate",
                "shape_batch_size": self._shape_batch_size,
                "max_parallel_shape_batches": self._max_parallel_batches,
                "shape_ids": [shape_id for batch in batches for shape_id in batch.shape_ids],
                "correct": evaluation.correct,
                "latency_us": evaluation.latency_us,
                "batches": [
                    {
                        "batch_index": batch.index,
                        "shape_ids": list(batch.shape_ids),
                        "agate_job_id": outcome.job_id,
                        "correct": outcome.evaluation.correct,
                        "latency_us": outcome.evaluation.latency_us,
                        "job": outcome.job,
                    }
                    for batch, outcome in zip(batches, outcomes, strict=True)
                ],
            },
        )
        return BatchedEvaluateOutcome(
            job,
            evaluation,
            None,
            _worker_result(evaluation, batches, outcomes),
            outcomes,
        )


def sorted_shape_ids(contract: AgateEvaluationContractV1) -> tuple[str, ...]:
    """Return the canonical numeric-then-lexicographic Shape order."""
    return tuple(
        sorted(
            contract.shapes,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
    )


def subset_evaluation_contract(
    contract: AgateEvaluationContractV1,
    shape_ids: tuple[str, ...],
) -> AgateEvaluationContractV1:
    """Copy a sealed contract while retaining only one ordered Shape batch."""
    return contract.model_copy(
        update={
            "shapes": {shape_id: contract.shapes[shape_id] for shape_id in shape_ids},
            "metadata": subset_shape_document(contract.metadata, shape_ids, metadata=True),
            "roofline": subset_shape_document(contract.roofline, shape_ids, metadata=False),
        }
    )


def subset_shape_document(
    value: dict[str, JsonValue] | None,
    shape_ids: tuple[str, ...] | list[str],
    *,
    metadata: bool,
) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    result = dict(value)
    per_shape = result.get("shapes")
    if isinstance(per_shape, dict):
        result["shapes"] = {
            shape_id: per_shape[shape_id] for shape_id in shape_ids if shape_id in per_shape
        }
    if metadata and "num_shapes" in result:
        result["num_shapes"] = len(shape_ids)
    return result


def _batch_idempotency_key(
    parent: str,
    index: int,
    shape_ids: tuple[str, ...],
) -> str:
    digest = hashlib.sha256(f"{parent}:{index}:{','.join(shape_ids)}".encode()).hexdigest()
    return f"shape-batch:{digest}"


def _aggregate_batch_evaluations(
    batches: tuple[ShapeBatch, ...],
    outcomes: tuple[ShapeBatchOutcome, ...],
) -> EvaluationV2:
    if any(not outcome.evaluation.correct for outcome in outcomes):
        return EvaluationV2(correct=False, latency_us=None)
    weighted_log_sum = 0.0
    shape_count = 0
    latencies: list[float] = []
    for batch, outcome in zip(batches, outcomes, strict=True):
        latency = outcome.evaluation.latency_us
        if latency is None or latency <= 0 or not math.isfinite(latency):
            return EvaluationV2(correct=False, latency_us=None)
        count = len(batch.shape_ids)
        latencies.append(latency)
        weighted_log_sum += count * math.log(latency)
        shape_count += count
    if len(set(latencies)) == 1:
        return EvaluationV2(correct=True, latency_us=latencies[0])
    return EvaluationV2(
        correct=True,
        latency_us=math.exp(weighted_log_sum / shape_count),
    )


def _worker_result(
    evaluation: EvaluationV2,
    batches: tuple[ShapeBatch, ...],
    outcomes: tuple[ShapeBatchOutcome, ...],
) -> JsonValue:
    latency_by_shape: dict[str, JsonValue] = {}
    correctness_values: list[object] = []
    for outcome in outcomes:
        worker = outcome.worker_result
        if not isinstance(worker, dict):
            continue
        values = worker.get("latency_us_by_shape")
        if isinstance(values, dict):
            latency_by_shape.update(values)
        correctness = worker.get("correctness")
        if isinstance(correctness, dict):
            correctness_values.append(correctness)
    return cast(
        JsonValue,
        {
            "all_pass": evaluation.correct,
            "correctness": merge_correctness_summaries(
                correctness_values,
                passed=evaluation.correct,
            ),
            "failures": (
                []
                if evaluation.correct
                else [
                    "one or more hidden evaluator cases failed; "
                    "reproduce within the public shape_domain"
                ]
            ),
            "latency_us_geomean": evaluation.latency_us if evaluation.correct else 0.0,
            "latency_us_by_shape": latency_by_shape,
            "shape_batch_count": len(batches),
            "shape_ids_are_opaque": True,
            "hidden_case_details": "shape inputs and failure details withheld",
        },
    )


def _candidate_rejection_detail(outcome: ShapeBatchOutcome) -> JsonValue | None:
    job = outcome.job
    if not isinstance(job, dict) or job.get("status") != "rejected":
        return None
    detail = job.get("error")
    return detail if detail is not None else "candidate request rejected"


def _candidate_rejected_outcome(
    batch: ShapeBatch,
    outcome: ShapeBatchOutcome,
    *,
    batch_count: int,
) -> BatchedEvaluateOutcome:
    detail = _candidate_rejection_detail(outcome)
    if detail is None:
        raise AssertionError("Candidate rejection outcome lost its rejection detail")
    error = cast(
        JsonValue,
        {
            "category": CANDIDATE_REJECTED_CATEGORY,
            "detail": detail,
        },
    )
    job = cast(
        JsonValue,
        {
            "schema_version": 1,
            "operation": "shape_batched_evaluate",
            "status": "rejected",
            "error": error,
            "rejected_batch_index": batch.index,
            "shape_batch_count": batch_count,
        },
    )
    worker_result = project_candidate_rejection(detail)
    return BatchedEvaluateOutcome(
        job,
        EvaluationV2(correct=False, latency_us=None),
        None,
        worker_result,
        (outcome,),
    )


__all__ = [
    "CANDIDATE_REJECTED_CATEGORY",
    "EVALUATE_MAX_PARALLEL_BATCHES",
    "EVALUATE_SHAPE_BATCH_SIZE",
    "BatchedEvaluateOutcome",
    "ShapeBatch",
    "ShapeBatchOutcome",
    "ShapeBatchedEvaluateExecutor",
    "sorted_shape_ids",
    "subset_evaluation_contract",
]
