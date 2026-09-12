"""Per-Shape repeated measurement aggregation tests."""

from __future__ import annotations

import pytest

from atrex_runtime.gateway.stability import (
    measurement_aggregation_summary,
    median_latency_by_shape,
    replace_shape_latencies,
)


def test_three_measurements_are_aggregated_by_per_shape_median() -> None:
    assert median_latency_by_shape(
        (
            {"0": 10.0, "1": 100.0},
            {"0": 30.0, "1": 80.0},
            {"0": 20.0, "1": 120.0},
        )
    ) == {"0": 20.0, "1": 100.0}
    assert measurement_aggregation_summary() == {
        "repetitions": 3,
        "method": "per_shape_median",
    }


def test_measurement_aggregation_requires_exactly_three_complete_shape_maps() -> None:
    with pytest.raises(ValueError, match="requires 3 samples"):
        median_latency_by_shape(({"0": 10.0}, {"0": 11.0}))
    with pytest.raises(ValueError, match="inconsistent Shape coverage"):
        median_latency_by_shape(({"0": 10.0}, {"0": 11.0}, {"1": 12.0}))


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
