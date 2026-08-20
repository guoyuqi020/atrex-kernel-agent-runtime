"""Ordinary Evaluate repetition aggregation tests."""

from __future__ import annotations

import pytest

from atrex_runtime.gateway.protocol import EvaluationV2
from atrex_runtime.gateway.repeated_evaluate import (
    aggregate_evaluations,
    repeated_evaluate_result,
    repeated_evaluate_worker_result,
)


def test_repeated_evaluate_requires_all_runs_and_uses_arithmetic_mean() -> None:
    evaluations = (
        EvaluationV2(correct=True, latency_us=9.0),
        EvaluationV2(correct=True, latency_us=12.0),
        EvaluationV2(correct=True, latency_us=15.0),
    )

    aggregate = aggregate_evaluations(evaluations)
    raw = repeated_evaluate_result(
        ({"job_id": "a"}, {"job_id": "b"}, {"job_id": "c"}),
        evaluations,
    )
    projected = repeated_evaluate_worker_result(evaluations)

    assert aggregate.correct is True
    assert aggregate.latency_us == pytest.approx(12.0)
    assert isinstance(raw, dict)
    assert raw["aggregation"] == "arithmetic_mean"
    assert raw["jobs"] == [{"job_id": "a"}, {"job_id": "b"}, {"job_id": "c"}]
    assert isinstance(projected, dict)
    assert projected["latency_us"] == pytest.approx(12.0)
    assert len(projected["measurements"]) == 3  # type: ignore[arg-type]


def test_repeated_evaluate_rejects_the_aggregate_when_one_run_fails() -> None:
    aggregate = aggregate_evaluations(
        (
            EvaluationV2(correct=True, latency_us=9.0),
            EvaluationV2(correct=False, latency_us=None),
        )
    )

    assert aggregate.correct is False
    assert aggregate.latency_us is None
