"""Safe aggregate correctness projections for Agent-visible Gateway results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..artifacts.local import JsonValue

_METRIC_ALIASES = {
    "rel_err": frozenset({"rel_err", "relative_l2"}),
    "max_abs_err": frozenset({"max_abs_err", "max_elementwise_abs_diff"}),
    "max_rel_err": frozenset({"max_rel_err", "max_elementwise_rel_diff"}),
}


def correctness_summary(value: object, *, passed: bool) -> dict[str, JsonValue]:
    """Project worst-case scalar errors without exposing private Shape or Case details."""
    payload = (
        value.get("correctness")
        if isinstance(value, Mapping) and "correctness" in value
        else value
    )
    metrics = {name: _maximum_metric(payload, aliases) for name, aliases in _METRIC_ALIASES.items()}
    return {
        "status": "PASS" if passed else "FAIL",
        "rel_err": metrics["rel_err"],
        "max_abs_err": metrics["max_abs_err"],
        "max_rel_err": metrics["max_rel_err"],
    }


def merge_correctness_summaries(
    values: Sequence[object],
    *,
    passed: bool,
) -> dict[str, JsonValue]:
    """Merge already-projected or raw correctness values using worst-case errors."""
    return correctness_summary({"correctness": list(values)}, passed=passed)


def _maximum_metric(value: object, aliases: frozenset[str]) -> float | None:
    maximum: float | None = None
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in aliases:
                number = _nonnegative_number(child)
                if number is not None:
                    maximum = number if maximum is None else max(maximum, number)
            nested = _maximum_metric(child, aliases)
            if nested is not None:
                maximum = nested if maximum is None else max(maximum, nested)
    elif isinstance(value, (list, tuple)):
        for child in value:
            nested = _maximum_metric(child, aliases)
            if nested is not None:
                maximum = nested if maximum is None else max(maximum, nested)
    return maximum


def _nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 and math.isfinite(number) else None


__all__ = ["correctness_summary", "merge_correctness_summaries"]
