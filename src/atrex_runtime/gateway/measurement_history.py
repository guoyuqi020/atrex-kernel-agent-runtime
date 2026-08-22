"""Normalize Agent-safe Evaluate and Profile results for cross-Attempt reuse."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..artifacts.local import JsonValue
from .control_models import GatewayMeasurementPoint, GatewayOperation

if TYPE_CHECKING:
    from .proxy import GatewayAdapterRequest, GatewayAdapterResult


def normalized_measurement_points(
    request: GatewayAdapterRequest,
    result: GatewayAdapterResult,
) -> tuple[GatewayMeasurementPoint, ...]:
    """Extract only bounded scalar facts that are already safe for an Agent to observe."""
    points: list[GatewayMeasurementPoint] = []
    if request.operation is GatewayOperation.EVALUATE:
        points.extend(_evaluation_points(result))
        if result.profile_result is not None:
            points.extend(
                _profile_points(
                    result.profile_result,
                    profile_level="sol",
                    requested_shape_id=_text(request.parameters.get("shape_id")),
                )
            )
    elif request.operation is GatewayOperation.PROFILE:
        points.extend(
            _profile_points(
                result.worker_result if result.worker_result is not None else result.result,
                profile_level=request.profile_level,
                requested_shape_id=_text(request.parameters.get("shape_id")),
            )
        )
    return tuple(points)


def _evaluation_points(result: GatewayAdapterResult) -> list[GatewayMeasurementPoint]:
    evaluation = result.evaluation
    if evaluation is None:
        return []
    aggregate: dict[str, JsonValue] = {"correct": evaluation.correct, "aggregate": True}
    if evaluation.latency_us is not None:
        aggregate["latency_us"] = evaluation.latency_us
    points = [
        GatewayMeasurementPoint(
            kind=GatewayOperation.EVALUATE,
            profile_level=None,
            shape_id=None,
            kernel_name=None,
            metrics=aggregate,
        )
    ]
    worker = result.worker_result
    if not isinstance(worker, dict):
        return points
    by_shape = worker.get("latency_us_by_shape")
    if isinstance(by_shape, dict):
        for shape_id in sorted(by_shape):
            latency = _number(by_shape.get(shape_id), positive=True)
            if latency is None:
                continue
            points.append(
                GatewayMeasurementPoint(
                    kind=GatewayOperation.EVALUATE,
                    profile_level=None,
                    shape_id=shape_id,
                    kernel_name=None,
                    metrics={"correct": True, "latency_us": latency},
                )
            )
    repetitions = worker.get("measurements")
    if isinstance(repetitions, list):
        for raw in repetitions:
            if not isinstance(raw, dict):
                continue
            repeat = raw.get("repeat")
            correct = raw.get("correct")
            if (
                isinstance(repeat, bool)
                or not isinstance(repeat, int)
                or not isinstance(correct, bool)
            ):
                continue
            metrics: dict[str, JsonValue] = {"correct": correct, "repeat": repeat}
            latency = _number(raw.get("latency_us"), positive=True)
            if latency is not None:
                metrics["latency_us"] = latency
            points.append(
                GatewayMeasurementPoint(
                    kind=GatewayOperation.EVALUATE,
                    profile_level=None,
                    shape_id=None,
                    kernel_name=None,
                    metrics=metrics,
                )
            )
    return points


def _profile_points(
    value: JsonValue,
    *,
    profile_level: str | None,
    requested_shape_id: str | None,
) -> list[GatewayMeasurementPoint]:
    if not isinstance(value, dict) or value.get("status") != "succeeded":
        return []
    result = value.get("result")
    if not isinstance(result, dict):
        return []
    raw_shape_id = result.get("shape_id")
    shape_id = raw_shape_id if isinstance(raw_shape_id, str) else requested_shape_id
    kernels = result.get("kernels")
    if not isinstance(kernels, list):
        return []
    points: list[GatewayMeasurementPoint] = []
    for raw in kernels:
        if not isinstance(raw, dict):
            continue
        metrics = _profile_metrics(raw)
        if not metrics:
            continue
        name = raw.get("name", raw.get("kernel_name"))
        points.append(
            GatewayMeasurementPoint(
                kind=GatewayOperation.PROFILE,
                profile_level=profile_level,
                shape_id=shape_id,
                kernel_name=name if isinstance(name, str) and name else None,
                metrics=metrics,
            )
        )
    return points


def _profile_metrics(raw: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    metrics: dict[str, JsonValue] = {}
    aliases = {
        "compute_sol_pct": "compute_sol_pct",
        "mem_sol_pct": "memory_sol_pct",
        "memory_sol_pct": "memory_sol_pct",
        "dram_pct": "dram_pct",
        "occupancy_pct": "occupancy_pct",
        "registers": "registers_per_thread",
        "registers_per_thread": "registers_per_thread",
        "shared_memory_bytes": "shared_memory_bytes",
        "smem_bytes": "shared_memory_bytes",
        "waves_per_sm": "waves_per_sm",
    }
    for source, target in aliases.items():
        value = _number(raw.get(source), positive=False)
        if value is not None:
            metrics[target] = value
    duration = _number(raw.get("duration"), positive=True)
    unit = raw.get("duration_unit")
    if duration is not None and isinstance(unit, str):
        scale = {"ns": 0.001, "us": 1.0, "ms": 1_000.0, "s": 1_000_000.0}.get(unit)
        if scale is not None:
            metrics["duration_us"] = duration * scale
    bound = raw.get("bound")
    if isinstance(bound, str) and bound:
        metrics["bound"] = bound
    traffic = raw.get("traffic")
    if isinstance(traffic, dict):
        for name in (
            "achieved_dram_gbps",
            "dram_bytes",
            "dram_bytes_read",
            "dram_bytes_write",
            "l2_bytes",
        ):
            value = _number(traffic.get(name), positive=False)
            if value is not None:
                metrics[name] = value
    return metrics


def _number(value: JsonValue | None, *, positive: bool) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        return None
    return number


def _text(value: JsonValue | None) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["normalized_measurement_points"]
