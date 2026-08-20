"""Shared Shape-batched ordinary Evaluate execution tests."""

from __future__ import annotations

import math

import anyio
import pytest

from atrex_runtime.gateway.batched_evaluate import (
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
async def test_shape_batch_executor_uses_four_shapes_and_four_workers() -> None:
    executor = ShapeBatchedEvaluateExecutor()
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

    result = await executor.run(_contract(4), "original-key", evaluate)

    assert result.job is raw
    assert result.job_id == "only"
    assert result.worker_result == {"all_pass": True}


@pytest.mark.anyio
async def test_shape_batch_executor_fails_closed_when_any_batch_is_incorrect() -> None:
    executor = ShapeBatchedEvaluateExecutor()

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
