"""Bounded Agate diagnostics and timing evidence, without evaluator input material.

Error text is upstream text, not a Runtime paraphrase. The surrounding fields only
identify which stage/opaque shape/output failed. Never include input generators,
sample seeds, input values or whole job/request documents in the Agent response.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import cast

from ..artifacts.local import JsonValue
from ..domain.errors import InfrastructureError
from .failures import infrastructure_detail

FEEDBACK_KEYS = ("diagnostics", "performance_by_shape", "feedback_truncation")
_MAX_DIAGNOSTICS = 128
_MAX_FEEDBACK_BYTES = 256 * 1024
_MAX_KERNELS = 64


def _count(value: JsonValue) -> int:
    return value if isinstance(value, int) else 0


def _text(value: object) -> str:
    return infrastructure_detail(InfrastructureError(str(value)))


def evaluation_feedback(raw: object, shape_ids: Iterable[str]) -> dict[str, JsonValue]:
    """Extract actionable failures and compact timing data from one native Eval result."""
    if not isinstance(raw, dict):
        return {}
    payload = raw.get("result", raw)
    if not isinstance(payload, dict):
        return {}
    allowed = set(shape_ids)
    diagnostics: list[JsonValue] = []
    performance: dict[str, JsonValue] = {}
    omitted = 0

    def add(stage: str, shape_id: str, error: object, **context: JsonValue) -> None:
        nonlocal omitted
        if not error:
            return
        if len(diagnostics) >= _MAX_DIAGNOSTICS:
            omitted += 1
            return
        original = str(error)
        message = _text(original)
        diagnostics.append(
            {
                "stage": stage,
                "shape_id": shape_id,
                **context,
                "error": message,
                **({"text_truncated": True} if len(original.encode()) > 8192 else {}),
            }
        )

    # Top-level error text is evidence too; never copy whole job/request objects.
    records = (
        (("job", raw), ("evaluation", payload))
        if payload is not raw
        else (("evaluation", payload),)
    )
    for stage, record in records:
        error = record.get("error")
        if isinstance(error, str):
            add(stage, "", error)
        elif isinstance(error, dict):
            add(stage, "", error.get("message") or error.get("reason"))

    for stage in ("compile", "correctness", "performance"):
        section = payload.get(stage)
        shapes = section.get("shapes", {}) if isinstance(section, dict) else {}
        if not isinstance(shapes, dict):
            continue
        for sid, value in shapes.items():
            if sid not in allowed or not isinstance(value, dict):
                continue
            add(stage, sid, value.get("error"))
            cases = value.get("cases", [])
            if isinstance(cases, list):
                for index, case in enumerate(cases):
                    if not isinstance(case, dict):
                        continue
                    add(stage, sid, case.get("error"), case_index=index)
                    for kind in ("outputs", "mutated_inputs", "unexpected_mutations"):
                        checks = case.get(kind, [])
                        if not isinstance(checks, list):
                            continue
                        for check in checks:
                            if not isinstance(check, dict) or check.get("passed") is not False:
                                continue
                            add(
                                stage,
                                sid,
                                check.get("error")
                                or "Agate reported a failed check without an error message",
                                case_index=index,
                                check=kind,
                                name=_text(check.get("name", "")),
                                **{
                                    k: check[k]
                                    for k in (
                                        "max_elementwise_abs_diff",
                                        "max_elementwise_rel_diff",
                                        "relative_l2",
                                    )
                                    if isinstance(check.get(k), (int, float))
                                },
                            )
            if stage == "performance":
                info: dict[str, JsonValue] = {
                    k: cast(JsonValue, value[k])
                    for k in (
                        "benchmark_mode",
                        "cache_flush_mb",
                        "capture_time_ms",
                    )
                    if isinstance(value.get(k), (str, int, float, bool))
                }
                stats = value.get("timing_stats")
                if isinstance(stats, dict):
                    info["timing_stats"] = {
                        k: v
                        for k, v in stats.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
                kernels = value.get("kernel_events")
                if isinstance(kernels, list):
                    info["kernel_events"] = [
                        {
                            k: (_text(v) if isinstance(v, str) else v)
                            for k, v in kernel.items()
                            if k in ("name", "calls", "device_time_us")
                            and isinstance(v, (str, int, float))
                        }
                        for kernel in kernels[:_MAX_KERNELS]
                        if isinstance(kernel, dict)
                    ]
                    if len(kernels) > _MAX_KERNELS:
                        info["omitted_kernel_events"] = len(kernels) - _MAX_KERNELS
                if info:
                    performance[sid] = info

    passed = payload.get("passed")
    if isinstance(passed, dict):
        for stage in ("compile", "correctness", "performance"):
            verdicts = passed.get(stage, {})
            if isinstance(verdicts, dict):
                for sid, verdict in verdicts.items():
                    if (
                        sid in allowed
                        and isinstance(verdict, dict)
                        and str(verdict.get("status", "")).lower() in ("failed", "fail", "error")
                        and not any(
                            isinstance(d, dict)
                            and d.get("shape_id") == sid
                            and d.get("stage") == stage
                            for d in diagnostics
                        )
                    ):
                        add(stage, sid, verdict.get("reason") or "Agate reported stage failure")
    return bound_feedback(
        {
            "diagnostics": diagnostics,
            "performance_by_shape": performance,
            **({"feedback_truncation": {"omitted_diagnostics": omitted}} if omitted else {}),
        }
    )


def bound_feedback(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Bound the whole feedback, keeping errors before optional timing detail."""
    raw_diagnostics = value.get("diagnostics")
    raw_performance = value.get("performance_by_shape")
    raw_truncation = value.get("feedback_truncation")
    diagnostics = list(raw_diagnostics) if isinstance(raw_diagnostics, list) else []
    performance = dict(raw_performance) if isinstance(raw_performance, dict) else {}
    truncation = dict(raw_truncation) if isinstance(raw_truncation, dict) else {}
    result: dict[str, JsonValue] = {"diagnostics": diagnostics, "performance_by_shape": performance}
    while len(json.dumps(result, ensure_ascii=False).encode()) > _MAX_FEEDBACK_BYTES:
        if performance:
            performance.pop(next(reversed(performance)))
            key = "omitted_performance_shapes"
        elif diagnostics:
            diagnostics.pop()
            key = "omitted_diagnostics"
        else:
            break
        truncation[key] = _count(truncation.get(key)) + 1
    if truncation:
        result["feedback_truncation"] = {
            **truncation,
            "reason": "Runtime response size limit; original errors retained first",
        }
    return result


def merge_feedback(values: Iterable[object]) -> dict[str, JsonValue]:
    """Carry projected feedback through batch aggregation without re-paraphrasing errors."""
    diagnostics: list[JsonValue] = []
    performance: dict[str, JsonValue] = {}
    omitted: dict[str, JsonValue] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        ds = value.get("diagnostics")
        if isinstance(ds, list):
            diagnostics.extend(ds)
        ps = value.get("performance_by_shape")
        if isinstance(ps, dict):
            performance.update(ps)
        truncation = value.get("feedback_truncation")
        if isinstance(truncation, dict):
            for key, count in truncation.items():
                if isinstance(count, int):
                    omitted[key] = _count(omitted.get(key)) + count
    return bound_feedback(
        {
            "diagnostics": diagnostics,
            "performance_by_shape": performance,
            **({"feedback_truncation": omitted} if omitted else {}),
        }
    )


def failure_notice(correct: bool) -> list[JsonValue]:
    return (
        []
        if correct
        else [
            "Evaluation failed. See diagnostics for Agate's original failure reasons; "
            "the input_scope field identifies whether this evaluated custom or contract inputs."
        ]
    )
