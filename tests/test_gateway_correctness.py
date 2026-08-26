"""Tests for safe aggregate correctness projections."""

from atrex_runtime.gateway.correctness import (
    correctness_summary,
    merge_correctness_summaries,
)


def test_correctness_summary_uses_worst_error_without_shape_details() -> None:
    value = {
        "correctness": {
            "shapes": {
                "0": {
                    "cases": [
                        {
                            "outputs": [
                                {
                                    "relative_l2": 0.001,
                                    "max_elementwise_abs_diff": 0.0009765625,
                                    "max_elementwise_rel_diff": 0.0078125,
                                }
                            ]
                        }
                    ]
                },
                "1": {
                    "cases": [
                        {
                            "outputs": [
                                {
                                    "relative_l2": 0.002,
                                    "max_elementwise_abs_diff": 0.0005,
                                    "max_elementwise_rel_diff": 0.006,
                                }
                            ]
                        }
                    ]
                },
            }
        }
    }

    assert correctness_summary(value, passed=True) == {
        "status": "PASS",
        "rel_err": 0.002,
        "max_abs_err": 0.0009765625,
        "max_rel_err": 0.0078125,
    }


def test_correctness_summary_preserves_failed_error_and_merges_worst_values() -> None:
    merged = merge_correctness_summaries(
        [
            {"rel_err": 0.01, "max_abs_err": 0.5, "max_rel_err": 0.1},
            {"rel_err": 0.02, "max_abs_err": 0.25, "max_rel_err": 0.2},
        ],
        passed=False,
    )

    assert merged == {
        "status": "FAIL",
        "rel_err": 0.02,
        "max_abs_err": 0.5,
        "max_rel_err": 0.2,
    }
