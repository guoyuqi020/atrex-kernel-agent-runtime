"""Aggregation primitives for independent ordinary Evaluate repetitions."""

from __future__ import annotations

import math
import statistics
from typing import cast

from ..artifacts.local import JsonValue
from .evaluation_feedback import FEEDBACK_KEYS, merge_feedback
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


def repeated_evaluate_worker_result(
    evaluations: tuple[EvaluationV2, ...],
    *,
    worker_results: tuple[JsonValue, ...] = (),
) -> JsonValue:
    """Preserve measurements and already-projected errors across repeat aggregation."""
    aggregate = aggregate_evaluations(evaluations)
    errors: list[JsonValue] = [
        {"repeat": repeat, "error": result["error"]}
        for repeat, result in enumerate(worker_results)
        if isinstance(result, dict) and result.get("error") is not None
    ]
    return cast(
        JsonValue,
        {
            "status": "succeeded" if aggregate.correct else "failed",
            "repeats": len(evaluations),
            "aggregation": "arithmetic_mean",
            "correct": aggregate.correct,
            "latency_us": aggregate.latency_us,
            **merge_feedback(worker_results),
            "measurements": [
                {
                    "repeat": repeat,
                    **_feedback(worker_results[repeat] if repeat < len(worker_results) else None),
                    "correct": evaluation.correct,
                    "latency_us": evaluation.latency_us,
                }
                for repeat, evaluation in enumerate(evaluations)
            ],
            "hidden_case_details": ("shape inputs withheld; upstream diagnostics preserved"),
            **(
                {"error": {"category": "evaluate_repetition_failed", "details": errors}}
                if errors else {}
            ),
        },
    )


def _feedback(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in FEEDBACK_KEYS if key in value}
