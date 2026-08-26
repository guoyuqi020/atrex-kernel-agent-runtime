"""Agent-safe projections of Gateway results backed by private evaluator cases."""

from __future__ import annotations

import math
import statistics

from ..artifacts.local import JsonValue
from .correctness import correctness_summary
from .protocol import EvaluationV2

_PRIVATE_RESULT_KEYS = frozenset(
    {
        "args",
        "arguments",
        "case",
        "cases",
        "init_kwargs",
        "inputs",
        "input_kwargs",
        "input_py",
        "kwargs",
        "log",
        "logs",
        "metadata",
        "payload",
        "reference",
        "reference_py",
        "request",
        "roofline",
        "shape",
        "shapes",
        "spec",
        "stderr",
        "stdout",
        "workload",
        "workloads",
    }
)
_HIDDEN_DETAIL = "shape inputs and failure details withheld"
_HIDDEN_FAILURE = (
    "one or more hidden evaluator cases failed; reproduce within the public shape_domain"
)
_CANDIDATE_REJECTED_CATEGORY = "candidate_rejected"


def project_private_evaluation(
    raw: JsonValue,
    evaluation: EvaluationV2,
    expected_shape_ids: tuple[str, ...],
) -> JsonValue:
    """Expose useful measurements without exposing exact cases or evaluator diagnostics."""
    payload = _evaluation_payload(raw)
    latency_by_shape = _latency_by_shape(payload, expected_shape_ids)
    return {
        "all_pass": evaluation.correct,
        "correctness": correctness_summary(payload, passed=evaluation.correct),
        "failures": [] if evaluation.correct else [_HIDDEN_FAILURE],
        "latency_us_geomean": evaluation.latency_us if evaluation.correct else 0.0,
        "latency_us_by_shape": latency_by_shape,
        "shape_ids_are_opaque": True,
        "hidden_case_details": _HIDDEN_DETAIL,
    }


def project_private_job(raw: JsonValue) -> JsonValue:
    """Keep safe job evidence while recursively removing private request/case material."""
    if isinstance(raw, dict) and raw.get("status") == "rejected":
        return project_candidate_rejection(raw.get("error"))
    if isinstance(raw, dict) and raw.get("status") in {"failed", "cancelled"}:
        return {
            "status": raw.get("status"),
            "job_id": raw.get("job_id"),
            "error": "gateway job did not complete; hidden-case details withheld",
        }
    return _strip_private_fields(raw)


def project_candidate_rejection(detail: JsonValue) -> JsonValue:
    """Expose pre-job source/schema validation while stripping private evaluator material.

    Agate classifies these responses before a GPU job or hidden correctness case is executed. The
    diagnostics are therefore actionable Candidate/source validation, not hidden-case outcomes.
    Recursive stripping still removes evaluator inputs, references, Shapes, logs, and payloads if
    an upstream response includes them unexpectedly.
    """
    safe_detail = _strip_private_fields(detail)
    return {
        "status": "rejected",
        "error": {
            "category": _CANDIDATE_REJECTED_CATEGORY,
            "message": "candidate request rejected before evaluation job creation",
            "details": safe_detail,
        },
    }


def _evaluation_payload(raw: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(raw, dict):
        return {}
    nested = raw.get("result")
    return nested if isinstance(nested, dict) else raw


def _latency_by_shape(
    payload: dict[str, JsonValue],
    expected_shape_ids: tuple[str, ...],
) -> dict[str, JsonValue]:
    direct = payload.get("latency_us_by_shape")
    if isinstance(direct, dict):
        return {
            shape_id: number
            for shape_id in expected_shape_ids
            if (number := _positive_number(direct.get(shape_id))) is not None
        }
    performance = payload.get("performance")
    if not isinstance(performance, dict):
        return {}
    shapes = performance.get("shapes")
    if not isinstance(shapes, dict):
        return {}
    latencies: dict[str, JsonValue] = {}
    for shape_id in expected_shape_ids:
        shape = shapes.get(shape_id)
        if not isinstance(shape, dict) or shape.get("error") is not None:
            continue
        samples = shape.get("samples")
        if not isinstance(samples, list):
            continue
        values = [
            number
            for sample in samples
            if isinstance(sample, dict)
            and (number := _positive_number(sample.get("end_to_end_time_ms"))) is not None
        ]
        if values:
            latencies[shape_id] = statistics.median(values) * 1000.0
    return latencies


def _strip_private_fields(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _strip_private_fields(child)
            for key, child in value.items()
            if key.lower() not in _PRIVATE_RESULT_KEYS
        }
    if isinstance(value, list):
        return [_strip_private_fields(child) for child in value]
    return value


def _positive_number(value: JsonValue | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 and math.isfinite(number) else None


__all__ = ["project_private_evaluation", "project_private_job"]
