"""Agent-safe projections of Gateway results backed by private evaluator cases."""

from __future__ import annotations

import math
import re
import statistics
from typing import cast

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
        "shape_valid",
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
_HIDDEN_JOB_FAILURE = "gateway job did not complete; hidden-case details withheld"
_CANDIDATE_REJECTED_CATEGORY = "candidate_rejected"
_EVALUATION_STAGES = ("compile", "correctness", "performance")
_MAX_FAILURE_TEXT = 400
_MAX_TRACEBACK_TEXT = 4_000
_MAX_UNESCAPE_ROUNDS = 4
_STAGE_VERDICT = re.compile(
    r'([A-Za-z_]+)":\s*\{"0":\s*\{"status":\s*"(\w+)",\s*"reason":\s*"([^"]*)"'
)
_CANDIDATE_TRACEBACK = re.compile(
    r'(File "[^"]*/candidate/[^"]*", line \d+, in \w+\n(?:.*\n)*?)'
    r"((?:[A-Za-z_.]*(?:Error|Exception))[^\n]*)"
)
_CANDIDATE_FRAME_PREFIX = re.compile(r'^File "[^"]*/candidate/', re.MULTILINE)


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
            "error": _project_job_failure(raw.get("error")),
        }
    return _strip_private_fields(raw)


def _project_job_failure(error: JsonValue) -> JsonValue:
    """Whitelist Agate's classification, the stage verdicts, and the candidate traceback."""
    if not isinstance(error, dict):
        return _HIDDEN_JOB_FAILURE
    details = error.get("details")
    details = details if isinstance(details, dict) else {}
    logs = _unescape(details.get("logs_tail"))
    projected: dict[str, JsonValue] = {}
    for key, source in (
        ("failure_origin", details.get("failure_origin")),
        ("failure_rule", details.get("failure_rule")),
        ("reason", error.get("reason")),
        ("trace_id", error.get("trace_id")),
    ):
        if isinstance(source, str) and source.strip():
            projected[key] = source[:_MAX_FAILURE_TEXT]
    stages = _evaluation_stages(logs)
    if stages:
        projected["stages"] = cast(JsonValue, stages)
    traceback = _candidate_traceback(logs)
    if traceback is not None:
        projected["candidate_traceback"] = traceback
    return projected or _HIDDEN_JOB_FAILURE


def _unescape(value: JsonValue) -> str:
    """Undo the nested JSON escaping Agate applies to its captured log tail."""
    if not isinstance(value, str):
        return ""
    text = value
    for _ in range(_MAX_UNESCAPE_ROUNDS):
        try:
            decoded = text.encode("utf-8", "surrogateescape").decode("unicode_escape", "replace")
        except (UnicodeDecodeError, UnicodeEncodeError):
            break
        if decoded == text:
            break
        text = decoded
    return text


def _evaluation_stages(logs: str) -> dict[str, JsonValue]:
    """Read each stage verdict, tolerating a stage key clipped by the log-tail window."""
    stages: dict[str, JsonValue] = {}
    for match in _STAGE_VERDICT.finditer(logs):
        key, status, reason = match.group(1), match.group(2), match.group(3)
        stage = next((name for name in _EVALUATION_STAGES if name.endswith(key)), None)
        if stage is None or stage in stages:
            continue
        verdict: dict[str, JsonValue] = {
            "status": status,
            "reason": reason[:_MAX_FAILURE_TEXT],
        }
        if key != stage:
            verdict["key_truncated"] = True
        stages[stage] = cast(JsonValue, verdict)
    return {name: stages[name] for name in _EVALUATION_STAGES if name in stages}


def _candidate_traceback(logs: str) -> str | None:
    """Return only the Agent's own frames, so evaluator internals stay hidden."""
    match = _CANDIDATE_TRACEBACK.search(logs)
    if match is None:
        return None
    frames = _CANDIDATE_FRAME_PREFIX.sub('File "candidate/', match.group(1), count=0)
    return f"{frames.rstrip()}\n{match.group(2)}".strip()[:_MAX_TRACEBACK_TEXT]


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
