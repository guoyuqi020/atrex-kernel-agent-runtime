"""Aggregation primitives for independent ordinary Evaluate repetitions."""

from __future__ import annotations

import math
import statistics
from typing import cast

from ..artifacts.local import JsonValue
from .protocol import EvaluationV2


def aggregate_evaluations(evaluations: tuple[EvaluationV2, ...]) -> EvaluationV2:
    """Require every repetition to pass and average their aggregate latencies."""
    if not evaluations:
        raise ValueError("ordinary Evaluate aggregation requires at least one repetition")
    latencies = [evaluation.latency_us for evaluation in evaluations]
    if not all(evaluation.correct for evaluation in evaluations) or any(
        value is None or value <= 0 or not math.isfinite(value) for value in latencies
    ):
        return EvaluationV2(correct=False, latency_us=None)
    return EvaluationV2(
        correct=True,
        latency_us=statistics.fmean(cast(list[float], latencies)),
    )


def repeated_evaluate_result(
    jobs: tuple[JsonValue, ...],
    evaluations: tuple[EvaluationV2, ...],
) -> JsonValue:
    """Build the durable raw aggregate while retaining every child Agate result."""
    aggregate = aggregate_evaluations(evaluations)
    return cast(
        JsonValue,
        {
            "schema_version": 1,
            "operation": "evaluate",
            "repeats": len(evaluations),
            "aggregation": "arithmetic_mean",
            "correct": aggregate.correct,
            "latency_us": aggregate.latency_us,
            "jobs": list(jobs),
        },
    )


def repeated_evaluate_worker_result(evaluations: tuple[EvaluationV2, ...]) -> JsonValue:
    """Expose bounded repeat measurements without private Shape inputs or failures."""
    aggregate = aggregate_evaluations(evaluations)
    return cast(
        JsonValue,
        {
            "status": "succeeded" if aggregate.correct else "failed",
            "repeats": len(evaluations),
            "aggregation": "arithmetic_mean",
            "correct": aggregate.correct,
            "latency_us": aggregate.latency_us,
            "measurements": [
                {
                    "repeat": repeat,
                    "correct": evaluation.correct,
                    "latency_us": evaluation.latency_us,
                }
                for repeat, evaluation in enumerate(evaluations)
            ],
            "hidden_case_details": "shape inputs and failure details withheld",
        },
    )
