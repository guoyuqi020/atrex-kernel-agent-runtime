"""Deterministic repeated measurement aggregation for performance results."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import cast

from ..artifacts.local import JsonValue

MEASUREMENT_REPETITIONS = 3


def latency_by_shape(value: object) -> dict[str, float]:
    """Return the finite positive per-Shape latency map from a public result."""
    if not isinstance(value, Mapping):
        return {}
    raw = value.get("latency_us_by_shape")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(shape_id): float(latency)
        for shape_id, latency in raw.items()
        if isinstance(shape_id, str)
        and isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(float(latency))
        and float(latency) > 0
    }


def median_latency_by_shape(samples: tuple[Mapping[str, float], ...]) -> dict[str, float]:
    """Return the per-Shape median from three complete, identically keyed samples."""
    if len(samples) != MEASUREMENT_REPETITIONS:
        raise ValueError(f"measurement aggregation requires {MEASUREMENT_REPETITIONS} samples")
    expected = set(samples[0])
    if not expected or any(set(sample) != expected for sample in samples[1:]):
        raise ValueError("repeated measurements have inconsistent Shape coverage")
    return {
        shape_id: statistics.median(sample[shape_id] for sample in samples)
        for shape_id in sorted(
            expected,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
    }


def replace_shape_latencies(
    value: Mapping[str, JsonValue],
    replacements: Mapping[str, float],
) -> dict[str, JsonValue]:
    """Replace selected Shape values and mechanically recompute aggregate latency fields."""
    updated = dict(value)
    current = latency_by_shape(value)
    accepted = {
        shape_id: float(latency)
        for shape_id, latency in replacements.items()
        if shape_id in current
        and isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(float(latency))
        and float(latency) > 0
    }
    current.update(accepted)
    updated["latency_us_by_shape"] = cast(JsonValue, current)
    if current and accepted:
        values = list(current.values())
        geomean = (
            values[0]
            if len(values) == 1
            else math.exp(statistics.fmean(math.log(item) for item in values))
        )
        arithmetic = statistics.fmean(values)
        for key in ("latency_us_geomean", "latency_us"):
            if key in value:
                updated[key] = geomean
        if "latency_us_arith_mean" in value:
            updated["latency_us_arith_mean"] = arithmetic
    return updated


def measurement_aggregation_summary() -> dict[str, JsonValue]:
    """Build the compact Agent-visible repeated measurement description."""
    return {
        "repetitions": MEASUREMENT_REPETITIONS,
        "method": "per_shape_median",
    }


__all__ = [
    "MEASUREMENT_REPETITIONS",
    "latency_by_shape",
    "measurement_aggregation_summary",
    "median_latency_by_shape",
    "replace_shape_latencies",
]
