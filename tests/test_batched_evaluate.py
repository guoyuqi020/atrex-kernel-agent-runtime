"""Shared Shape-batched ordinary Evaluate execution tests."""

from __future__ import annotations

import math

import anyio
import pytest

from atrex_runtime.gateway.batched_evaluate import (
    CANDIDATE_REJECTED_CATEGORY,
    ShapeBatch,
    ShapeBatchedEvaluateExecutor,
    ShapeBatchOutcome,
)
from atrex_runtime.gateway.contract import (
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
)
from atrex_runtime.gateway.protocol import EvaluationV2


def _contract(count: int) -> AgateEvaluationContractV1:
    shapes = {str(index): [index] for index in range(count)}
    return AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py="reference",
        input_py="inputs",
        shapes=shapes,
        metadata={"num_shapes": count, "shapes": shapes, "shared": "metadata"},
        roofline={"shapes": shapes, "shared": "roofline"},
        options=AgateEvaluationOptionsV1(
            num_correctness_cases=1,
            bench_iters=5,
            atol=0,
            rtol=0,
            timeout_s=60,
        ),
        lock_clocks=True,
    )


@pytest.mark.anyio
async def test_shape_batch_executor_honours_the_requested_split_and_worker_cap() -> None:
    executor = ShapeBatchedEvaluateExecutor(shape_batch_size=4, max_parallel_batches=4)
    active = 0
    peak = 0
    seen: list[ShapeBatch] = []
    lock = anyio.Lock()

    async def evaluate(batch: ShapeBatch) -> ShapeBatchOutcome:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
            seen.append(batch)
        await anyio.sleep(0.01)
        async with lock:
            active -= 1
        latency = float(batch.index + 1)
        return ShapeBatchOutcome(
            {"status": "succeeded", "batch": batch.index},
            EvaluationV2(correct=True, latency_us=latency),
            f"job-{batch.index}",
            {"latency_us_by_shape": {shape_id: latency for shape_id in batch.shape_ids}},
        )

    result = await executor.run(_contract(20), "logical-evaluate", evaluate)

    assert peak == 4
    assert [batch.shape_ids for batch in seen] == [
        ("0", "1", "2", "3"),
        ("4", "5", "6", "7"),
        ("8", "9", "10", "11"),
        ("12", "13", "14", "15"),
        ("16", "17", "18", "19"),
    ]
    assert all(len(batch.contract.shapes) == 4 for batch in seen)
    assert all(batch.contract.metadata["num_shapes"] == 4 for batch in seen)  # type: ignore[index]
    assert len({batch.idempotency_key for batch in seen}) == 5
    assert result.evaluation.correct is True
    assert result.evaluation.latency_us == pytest.approx(
        math.exp(sum(4 * math.log(value) for value in (1, 2, 3, 4, 5)) / 20)
    )
    assert result.job_id is None
    assert isinstance(result.job, dict)
    assert result.job["operation"] == "shape_batched_evaluate"
    assert result.job["shape_batch_size"] == 4
    assert result.job["max_parallel_shape_batches"] == 4


@pytest.mark.anyio
async def test_shape_batch_executor_preserves_single_job_result() -> None:
    executor = ShapeBatchedEvaluateExecutor()
    raw = {"job_id": "only", "status": "succeeded"}

    async def evaluate(batch: ShapeBatch) -> ShapeBatchOutcome:
        assert batch.idempotency_key == "original-key"
        return ShapeBatchOutcome(
            raw,
            EvaluationV2(correct=True, latency_us=3.0),
            "only",
            {"all_pass": True},
        )

    result = await executor.run(_contract(1), "original-key", evaluate)

    assert result.job is raw
    assert result.job_id == "only"
    assert result.worker_result == {"all_pass": True}


@pytest.mark.anyio
async def test_shape_batch_executor_fails_closed_when_any_batch_is_incorrect() -> None:
    executor = ShapeBatchedEvaluateExecutor(shape_batch_size=4)

    async def evaluate(batch: ShapeBatch) -> ShapeBatchOutcome:
        correct = batch.index == 0
        return ShapeBatchOutcome(
            {"status": "succeeded", "batch": batch.index},
            EvaluationV2(
                correct=correct,
                latency_us=8.0 if correct else None,
            ),
        )

    result = await executor.run(_contract(5), "fail-closed", evaluate)

    assert result.evaluation == EvaluationV2(correct=False, latency_us=None)
    assert isinstance(result.job, dict)
    assert result.job["correct"] is False
    assert result.worker_result["all_pass"] is False  # type: ignore[index]


@pytest.mark.anyio
async def test_shape_batch_executor_stops_on_candidate_rejection() -> None:
    executor = ShapeBatchedEvaluateExecutor(shape_batch_size=4, max_parallel_batches=1)
    seen: list[int] = []
    rejection = {
        "reason": "candidate_validation_failed",
        "details": {"forbidden_imports": ["os"]},
    }

    async def evaluate(batch: ShapeBatch) -> ShapeBatchOutcome:
        seen.append(batch.index)
        if batch.index == 0:
            return ShapeBatchOutcome(
                {"status": "rejected", "error": rejection},
                EvaluationV2(correct=False, latency_us=None),
            )
        raise AssertionError("Candidate rejection must cancel pending Shape batches")

    result = await executor.run(_contract(12), "candidate-rejected", evaluate)

    assert seen == [0]
    assert result.evaluation == EvaluationV2(correct=False, latency_us=None)
    assert result.job_id is None
    assert result.batches[0].job == {"status": "rejected", "error": rejection}
    assert result.job == {
        "schema_version": 1,
        "operation": "shape_batched_evaluate",
        "status": "rejected",
        "error": {
            "category": CANDIDATE_REJECTED_CATEGORY,
            "detail": rejection,
        },
        "rejected_batch_index": 0,
        "shape_batch_count": 3,
    }
    assert result.worker_result == {
        "status": "rejected",
        "error": {
            "category": CANDIDATE_REJECTED_CATEGORY,
            "message": "candidate request rejected before evaluation job creation",
            "details": rejection,
        },
    }


@pytest.mark.anyio
async def test_default_matches_abba_one_shape_and_sixteen_workers() -> None:
    from atrex_runtime.config import SameAllocationAbbaComparisonSettings

    executor = ShapeBatchedEvaluateExecutor()
    seen: list[ShapeBatch] = []
    active = 0
    peak = 0

    async def evaluate(batch: ShapeBatch) -> ShapeBatchOutcome:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        seen.append(batch)
        await anyio.sleep(0.01)
        active -= 1
        return ShapeBatchOutcome(
            {"job_id": f"job-{batch.index}", "status": "succeeded"},
            EvaluationV2(correct=True, latency_us=5.0),
            f"job-{batch.index}",
            {"all_pass": True},
        )

    result = await executor.run(_contract(32), "logical-eval", evaluate)

    abba = SameAllocationAbbaComparisonSettings(method="same_allocation_abba")
    assert len(seen) == 32
    assert peak == abba.max_parallel_shape_batches == 16
    assert all(len(batch.shape_ids) == abba.shape_batch_size == 1 for batch in seen)
    assert {batch.shape_ids[0] for batch in seen} == {str(index) for index in range(32)}
    assert len({batch.idempotency_key for batch in seen}) == 32
    assert all(batch.contract.lock_clocks for batch in seen)
    assert all(batch.contract.metadata["num_shapes"] == 1 for batch in seen)
    assert all(set(batch.contract.roofline["shapes"]) == set(batch.shape_ids) for batch in seen)
    assert result.job_id is None
    assert result.evaluation.latency_us == pytest.approx(5)


@pytest.mark.anyio
async def test_explicit_unsplit_executor_still_uses_one_job() -> None:
    seen: list[ShapeBatch] = []

    async def evaluate(batch: ShapeBatch) -> ShapeBatchOutcome:
        seen.append(batch)
        return ShapeBatchOutcome({}, EvaluationV2(correct=True, latency_us=5))

    await ShapeBatchedEvaluateExecutor(shape_batch_size=None).run(_contract(14), "key", evaluate)
    assert len(seen) == 1
    assert len(seen[0].shape_ids) == 14
