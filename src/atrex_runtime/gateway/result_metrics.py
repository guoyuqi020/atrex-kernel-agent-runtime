"""Read display metrics from immutable Gateway result Artifacts."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.ids import ArtifactDigest, parse_artifact_digest
from .correctness import correctness_summary, merge_correctness_summaries


@dataclass(frozen=True, slots=True)
class GatewaySolSummary:
    """Best available SOL metric and the evidence path that produced it."""

    percent: float | None
    source: str | None
    detail: str | None = None


def gateway_result_projection(
    artifacts: LocalArtifactStore,
    digest: ArtifactDigest,
    *,
    correct: bool,
    latency_us: float | None,
) -> dict[str, JsonValue]:
    """Return an Agent-safe projection of one authoritative Gateway result."""
    value = _gateway_result_value(artifacts, digest)
    operation = value.get("operation")
    status = value.get("status")
    by_shape = _latency_by_shape_value(artifacts, value, seen={digest})
    return {
        "operation": operation if isinstance(operation, str) and operation else "evaluate",
        "status": status if isinstance(status, str) and status else "completed",
        "correct": correct,
        "correctness": _correctness_projection(artifacts, digest, passed=correct, seen=set()),
        "latency_us_geomean": latency_us,
        "latency_us_arith_mean": statistics.fmean(by_shape.values()) if by_shape else None,
        "latency_us_by_shape": cast(dict[str, JsonValue], by_shape),
    }


def _gateway_result_value(
    artifacts: LocalArtifactStore,
    digest: ArtifactDigest,
) -> dict[str, JsonValue]:
    stored = artifacts.verify(digest)
    if stored.kind is not ArtifactKind.GATEWAY_RESULT:
        raise ValueError("Kernel evaluation does not reference a Gateway Result")
    value = _read_json_object(stored.payload_path / "value.json")
    if value is None:
        raise ValueError("Gateway Result is not valid JSON")
    return value


def _correctness_projection(
    artifacts: LocalArtifactStore,
    digest: ArtifactDigest,
    *,
    passed: bool,
    seen: set[ArtifactDigest],
) -> dict[str, JsonValue]:
    if digest in seen:
        raise ValueError("Gateway Result measurement graph contains a cycle")
    visited = {*seen, digest}
    value = _gateway_result_value(artifacts, digest)
    operation = value.get("operation")
    if operation == "same_allocation_abba":
        return correctness_summary(value.get("candidate"), passed=passed)
    if operation == "evaluate_comparison":
        measurements = value.get("measurements")
        summaries: list[object] = []
        if isinstance(measurements, list):
            for measurement in measurements:
                if not isinstance(measurement, dict):
                    continue
                raw_digest = measurement.get("gateway_result_digest")
                if isinstance(raw_digest, str):
                    summaries.append(
                        _correctness_projection(
                            artifacts,
                            parse_artifact_digest(raw_digest),
                            passed=passed,
                            seen=visited,
                        )
                    )
        return merge_correctness_summaries(summaries, passed=passed)
    return correctness_summary(value, passed=passed)


def _latency_by_shape_digest(
    artifacts: LocalArtifactStore,
    digest: ArtifactDigest,
    *,
    seen: set[ArtifactDigest],
) -> dict[str, float]:
    if digest in seen:
        raise ValueError("Gateway Result measurement graph contains a cycle")
    return _latency_by_shape_value(
        artifacts,
        _gateway_result_value(artifacts, digest),
        seen={*seen, digest},
    )


def _latency_by_shape_value(
    artifacts: LocalArtifactStore,
    value: Mapping[str, object],
    *,
    seen: set[ArtifactDigest],
) -> dict[str, float]:
    operation = value.get("operation")
    if operation == "same_allocation_abba":
        candidate = value.get("candidate")
        if isinstance(candidate, Mapping):
            return _shape_mapping(candidate.get("latency_us_by_shape"))
    if operation == "evaluate_comparison":
        measurements = value.get("measurements")
        if isinstance(measurements, list):
            children = [
                _latency_by_shape_digest(
                    artifacts,
                    parse_artifact_digest(raw_digest),
                    seen=set(seen),
                )
                for measurement in measurements
                if isinstance(measurement, Mapping)
                and isinstance((raw_digest := measurement.get("gateway_result_digest")), str)
            ]
            return _mean_shape_mappings(children)
    direct = _shape_mapping(value.get("latency_us_by_shape"))
    if direct:
        return direct
    for key in ("result", "worker_result", "job"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            projected = _latency_by_shape_value(artifacts, nested, seen=set(seen))
            if projected:
                return projected
    stages = value.get("completed_stages")
    if isinstance(stages, list):
        for stage in reversed(stages):
            if isinstance(stage, Mapping):
                projected = _latency_by_shape_value(artifacts, stage, seen=set(seen))
                if projected:
                    return projected
    jobs = value.get("jobs")
    if isinstance(jobs, list):
        return _mean_shape_mappings(
            [
                projected
                for job in jobs
                if isinstance(job, Mapping)
                and (projected := _latency_by_shape_value(artifacts, job, seen=set(seen)))
            ]
        )
    return {}


def _shape_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(shape_id): number
        for shape_id, raw in value.items()
        if (number := _positive_finite_number(raw)) is not None
    }


def _positive_finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 and math.isfinite(number) else None


def _mean_shape_mappings(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    shared = set(values[0])
    for value in values[1:]:
        shared.intersection_update(value)
    return {
        shape_id: statistics.fmean(value[shape_id] for value in values)
        for shape_id in sorted(shared, key=_shape_sort_key)
    }


def _shape_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def gateway_result_sol_percent(
    artifacts: LocalArtifactStore,
    digest: ArtifactDigest,
) -> float | None:
    """Return the all-shape SOL percentage geomean, or None when it is unavailable."""
    return gateway_result_sol_summary(artifacts, digest).percent


def gateway_result_sol_summary(
    artifacts: LocalArtifactStore,
    digest: ArtifactDigest,
) -> GatewaySolSummary:
    """Return the SOL metric plus whether Roofline or NCU Profile supplied it."""
    try:
        stored = artifacts.verify(digest)
    except (OSError, ValueError) as error:
        return GatewaySolSummary(None, None, f"Gateway result unavailable: {error}")
    if stored.kind is not ArtifactKind.GATEWAY_RESULT:
        return GatewaySolSummary(None, None, "Artifact is not a Gateway result")
    value = _read_json_object(stored.payload_path / "value.json")
    repeated = _ordinary_evaluate_aggregate_sol(artifacts, value)
    if repeated is not None:
        return repeated
    abba = _same_allocation_abba_sol(value)
    if abba is not None:
        return abba
    roofline = _roofline_sol_percent(value)
    if roofline is not None:
        return GatewaySolSummary(roofline, "roofline")

    profile_path = stored.payload_path / "profile.json"
    if not profile_path.is_file():
        return GatewaySolSummary(None, None, "no Roofline or NCU Profile result")
    profile = _read_json_object(profile_path)
    profile_sol = _profile_sol_percent(profile)
    if profile_sol is not None:
        return GatewaySolSummary(profile_sol, "ncu-profile")
    return GatewaySolSummary(None, "ncu-profile", _profile_failure_detail(profile))


def _same_allocation_abba_sol(
    value: dict[str, JsonValue] | None,
) -> GatewaySolSummary | None:
    if value is None or value.get("operation") != "same_allocation_abba":
        return None
    candidate = value.get("candidate")
    if not isinstance(candidate, dict):
        return GatewaySolSummary(None, None, "ABBA result has no Candidate metrics")
    percentage = _nonnegative_finite_number(candidate.get("sol_pct"))
    if percentage is None:
        return GatewaySolSummary(
            None,
            None,
            "ABBA Candidate measurements contain no Roofline SOL",
        )
    return GatewaySolSummary(
        percentage,
        "roofline",
        "same-allocation ABBA Candidate aggregate",
    )


def _ordinary_evaluate_aggregate_sol(
    artifacts: LocalArtifactStore,
    value: dict[str, JsonValue] | None,
) -> GatewaySolSummary | None:
    if value is None or value.get("operation") != "evaluate_comparison":
        return None
    measurements = value.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        return GatewaySolSummary(None, None, "ordinary Evaluate aggregate has no measurements")
    percentages: list[float] = []
    sources: set[str] = set()
    for measurement in measurements:
        if not isinstance(measurement, dict):
            return GatewaySolSummary(None, None, "ordinary Evaluate measurement is invalid")
        raw_digest = measurement.get("gateway_result_digest")
        if not isinstance(raw_digest, str):
            return GatewaySolSummary(None, None, "ordinary Evaluate measurement has no result")
        try:
            digest = ArtifactDigest(str(parse_artifact_digest(raw_digest)))
        except ValueError:
            return GatewaySolSummary(None, None, "ordinary Evaluate result Digest is invalid")
        summary = gateway_result_sol_summary(artifacts, digest)
        if summary.percent is None or summary.source is None:
            return GatewaySolSummary(None, summary.source, summary.detail)
        percentages.append(summary.percent)
        sources.add(summary.source)
    if any(percentage == 0 for percentage in percentages):
        percent = 0.0
    else:
        percent = len(percentages) / sum(1.0 / percentage for percentage in percentages)
    source = next(iter(sources)) if len(sources) == 1 else "mixed"
    return GatewaySolSummary(percent, source, "ordinary Evaluate arithmetic-mean aggregate")


def _roofline_sol_percent(value: dict[str, JsonValue] | None) -> float | None:
    if value is None:
        return None
    if value.get("operation") == "bootstrap_staged_evaluate":
        stages = value.get("completed_stages")
        source_stage = value.get("latency_source_stage")
        if (
            not isinstance(stages, list)
            or not isinstance(source_stage, int)
            or isinstance(source_stage, bool)
            or source_stage < 0
            or source_stage >= len(stages)
        ):
            return None
        stage = stages[source_stage]
        if not isinstance(stage, dict):
            return None
        job = stage.get("job")
        return _roofline_sol_percent(job if isinstance(job, dict) else None)
    repeated_jobs = value.get("jobs")
    if value.get("aggregation") == "arithmetic_mean" and isinstance(repeated_jobs, list):
        repeated_percentages: list[float] = []
        for job in repeated_jobs:
            if not isinstance(job, dict):
                return None
            percentage = _roofline_sol_percent(job)
            if percentage is None:
                return None
            repeated_percentages.append(percentage)
        if not repeated_percentages:
            return None
        if any(percentage == 0 for percentage in repeated_percentages):
            return 0.0
        # SOL is inversely proportional to measured latency. The aggregate uses an
        # arithmetic latency mean, so its corresponding SOL is the harmonic mean.
        return len(repeated_percentages) / sum(
            1.0 / percentage for percentage in repeated_percentages
        )
    result = value.get("result")
    if not isinstance(result, dict):
        return None
    performance = result.get("performance")
    if not isinstance(performance, dict):
        return None
    shapes = performance.get("shapes")
    if not isinstance(shapes, dict) or not shapes:
        return None
    percentages: list[float] = []
    for shape in shapes.values():
        if not isinstance(shape, dict):
            return None
        sol = shape.get("sol")
        if not isinstance(sol, dict):
            return None
        percentage = _nonnegative_finite_number(sol.get("pct"))
        if percentage is None:
            return None
        percentages.append(percentage)
    if not percentages:
        return None
    if len(percentages) == 1:
        return percentages[0]
    if any(value == 0 for value in percentages):
        return 0.0
    return math.exp(sum(math.log(value) for value in percentages) / len(percentages))


def _profile_sol_percent(value: dict[str, JsonValue] | None) -> float | None:
    if value is None or value.get("status") != "succeeded":
        return None
    result = value.get("result")
    if not isinstance(result, dict):
        return None
    kernels = result.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        return None
    weighted = 0.0
    total_duration_ns = 0.0
    for kernel in kernels:
        if not isinstance(kernel, dict):
            continue
        compute = _nonnegative_finite_number(kernel.get("compute_sol_pct"))
        memory = _nonnegative_finite_number(kernel.get("mem_sol_pct"))
        duration = _nonnegative_finite_number(kernel.get("duration"))
        if compute is None or memory is None or duration is None or duration == 0:
            continue
        duration_ns = _duration_ns(duration, kernel.get("duration_unit"))
        if duration_ns is None:
            continue
        weighted += max(compute, memory) * duration_ns
        total_duration_ns += duration_ns
    return None if total_duration_ns == 0 else weighted / total_duration_ns


def _duration_ns(duration: float, unit: JsonValue | None) -> float | None:
    scale = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}
    return None if not isinstance(unit, str) or unit not in scale else duration * scale[unit]


def _profile_failure_detail(value: dict[str, JsonValue] | None) -> str:
    if value is None:
        return "invalid NCU Profile result"
    error = value.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    status = value.get("status")
    return f"NCU Profile {status}" if isinstance(status, str) else "NCU Profile unavailable"


def _read_json_object(path: Path) -> dict[str, JsonValue] | None:
    try:
        value: object = json.loads(path.read_bytes())
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _nonnegative_finite_number(value: JsonValue | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 and math.isfinite(number) else None


__all__ = ["GatewaySolSummary", "gateway_result_sol_percent", "gateway_result_sol_summary"]
