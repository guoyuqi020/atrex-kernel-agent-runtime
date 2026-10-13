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
        "shape_test",
        "test_shapes",
        "validation_shape_ids",
        "shape_split",
        "source_shape_ids",
        "test_shape_ids",
        "test_observation",
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


def project_source_tree_failure(raw: object) -> dict[str, JsonValue]:
    """Expose driver/child failures without forwarding private evaluator output."""
    diagnostic: dict[str, JsonValue] = {}
    if not isinstance(raw, dict):
        return {"reason": "source-tree evaluator returned an invalid payload"}
    if raw.get("error") is not None:
        diagnostic["driver_error"] = _source_failure_message(raw["error"])
    runs = raw.get("runs")
    if not isinstance(runs, list):
        diagnostic["reason"] = "source-tree evaluator returned no run records"
        return diagnostic
    diagnostic["run_count"] = len(runs)
    children: list[JsonValue] = []
    for index, run in enumerate(runs[:16]):
        if not isinstance(run, dict):
            continue
        child: dict[str, JsonValue] = {"run_index": index}
        code = run.get("exit_code")
        if isinstance(code, int) and not isinstance(code, bool):
            child["exit_code"] = code
        result = run.get("result")
        if isinstance(result, dict) and result.get("error") is not None:
            child["error"] = _source_failure_message(result["error"])
        if not isinstance(result, dict) or not isinstance(result.get("raw_result"), dict):
            child["reason"] = "evaluator child produced no structured evaluation result"
        # Only stage verdicts and candidate frames may escape the child's logs.
        projected = _project_job_failure({"details": {"logs_tail": run.get("stderr_tail")}})
        if isinstance(projected, dict):
            child.update(projected)
        stderr = _unescape(run.get("stderr_tail"))
        exception = re.search(r"^([\w.]+(?:Error|Exception):[^\n]*)$", stderr, re.MULTILINE)
        if exception is not None:
            child["exception"] = _source_failure_message(exception.group(1))
        elif stderr.strip() == "evaluation timed out":
            child["exception"] = "evaluation timed out"
        compiler_errors: list[JsonValue] = []
        for line in stderr.splitlines():
            if re.search(
                r"(?:\.(?:cu|cuh|cc|cpp)(?:\(|:\d)|\b(?:nvcc|ptxas)\b).*?\b(?:error|fatal)\b",
                line,
            ):
                line = re.sub(r"/(?:[^/\s:()]+/)+([^/\s:()]+)", r"\1", line)
                compiler_errors.append(_source_failure_message(line.strip()))
                if len(compiler_errors) == 8:
                    break
        if compiler_errors:
            child["compiler_errors"] = compiler_errors
        children.append(child)
    if children:
        diagnostic["runs"] = children
    return diagnostic


def _source_failure_message(value: object) -> JsonValue:
    if isinstance(value, dict):
        projected = _project_job_failure(value)
        message = value.get("message")
        if isinstance(message, str):
            if not isinstance(projected, dict):
                projected = {}
            projected["message"] = _source_failure_message(message)
        return projected
    if not isinstance(value, str):
        return "evaluator execution failed; private details withheld"
    # Exception messages may embed a private input dump rather than a code error.
    # Preserve the exception type in that case, never that input dump.
    if re.search(
        r"(?i)(?:hidden|private)[ _-]*(?:input|case|shape|reference|payload)|"
        r"""manual_seed|["'](?:input_py|reference_py|inputs|kwargs|shapes)["']\s*:""",
        value,
    ):
        exception = re.match(r"([\w.]+(?:Error|Exception)):", value)
        prefix = f"{exception.group(1)}: " if exception else ""
        return prefix + "evaluator execution failed; private details withheld"
    # Named input/seed values can also appear in an otherwise useful error sentence.
    message = re.sub(
        r"\b[A-Za-z_]\w*\s*=\s*(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|"
        r"\[[^\n]*\]|\{[^\n]*\}|\([^\n]*\))",
        lambda match: match.group(0).split("=", 1)[0] + "=[withheld]",
        value,
    )
    return message[:_MAX_FAILURE_TEXT]


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
        return project_job_failure(raw)
    return _strip_private_fields(raw)


def project_job_failure(job: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Project a failed execution even when its outer Job status says succeeded."""
    diagnostic: dict[str, JsonValue] = {
        "status": job.get("status"),
        "job_id": job.get("job_id"),
        "error": _project_job_failure(job.get("error")),
    }
    if isinstance(job.get("command_ok"), bool):
        diagnostic["command_ok"] = job["command_ok"]
    return diagnostic


def project_compile_job(raw: JsonValue) -> JsonValue:
    """Keep the compile verdict an Agent can act on and drop Agate's null envelope.

    An allowlist rather than a denylist, so a request echo Agate may add later, such as
    the `include_spec` digest, cannot reach the Agent by default.
    """
    if not isinstance(raw, dict):
        return _strip_private_fields(raw)
    if raw.get("status") == "rejected":
        return project_candidate_rejection(raw.get("error"))
    return {
        "status": raw.get("status"),
        "job_id": raw.get("job_id"),
        "error": _project_job_failure(raw.get("error")) if raw.get("error") else None,
        "result": _strip_private_fields(raw.get("result")),
    }


def _project_job_failure(error: JsonValue) -> JsonValue:
    """Whitelist Agate's classification, the stage verdicts, and the candidate traceback."""
    if not isinstance(error, dict):
        return _HIDDEN_JOB_FAILURE
    details = error.get("details")
    details = details if isinstance(details, dict) else {}
    logs = _unescape(details.get("logs_tail"))
    projected: dict[str, JsonValue] = {}
    for key, source in (
        ("error_class", error.get("error_class")),
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


__all__ = [
    "project_compile_job",
    "project_job_failure",
    "project_private_evaluation",
    "project_private_job",
]
