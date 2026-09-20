"""Per-Shape single measurement summary tests."""

from __future__ import annotations

import pytest

from atrex_runtime.gateway.stability import (
    measurement_aggregation_summary,
    measurement_values_by_shape,
    replace_shape_latencies,
)


def test_single_measurement_preserves_per_shape_values() -> None:
    assert measurement_values_by_shape(({"0": 10.0, "1": 100.0},)) == {
        "0": 10.0,
        "1": 100.0,
    }
    assert measurement_aggregation_summary() == {
        "repetitions": 1,
        "method": "single_measurement",
    }


def test_measurement_summary_requires_exactly_one_nonempty_shape_map() -> None:
    with pytest.raises(ValueError, match="requires exactly one sample"):
        measurement_values_by_shape(({"0": 10.0}, {"0": 11.0}))
    with pytest.raises(ValueError, match="empty Shape coverage"):
        measurement_values_by_shape(({},))


def test_replacing_shape_recomputes_aggregate_latencies() -> None:
    updated = replace_shape_latencies(
        {
            "latency_us_by_shape": {"0": 10.0, "1": 40.0},
            "latency_us_geomean": 20.0,
            "latency_us_arith_mean": 25.0,
        },
        {"1": 90.0},
    )

    assert updated["latency_us_by_shape"] == {"0": 10.0, "1": 90.0}
    assert updated["latency_us_geomean"] == pytest.approx(30.0)
    assert updated["latency_us_arith_mean"] == pytest.approx(50.0)
