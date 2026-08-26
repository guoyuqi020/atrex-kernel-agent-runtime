"""Gateway-result display metric extraction tests."""

from __future__ import annotations

import math
from pathlib import Path

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.gateway.result_metrics import (
    gateway_result_sol_percent,
    gateway_result_sol_summary,
)


def test_sol_percent_uses_all_shape_geometric_mean(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    result = artifacts.put_json(
        {
            "result": {
                "performance": {
                    "shapes": {
                        "0": {"sol": {"pct": 25.0, "reason": None}},
                        "1": {"sol": {"pct": 100.0, "reason": None}},
                    }
                }
            }
        },
        ArtifactKind.GATEWAY_RESULT,
    )

    assert math.isclose(gateway_result_sol_percent(artifacts, result) or 0.0, 50.0)


def test_sol_percent_is_absent_when_any_shape_has_no_roofline(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    result = artifacts.put_json(
        {
            "result": {
                "performance": {
                    "shapes": {
                        "0": {"sol": {"pct": 80.0, "reason": None}},
                        "1": {"sol": {"pct": None, "reason": "no roofline"}},
                    }
                }
            }
        },
        ArtifactKind.GATEWAY_RESULT,
    )

    assert gateway_result_sol_percent(artifacts, result) is None


def test_repeated_evaluate_sol_matches_arithmetic_latency_aggregation(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    result = artifacts.put_json(
        {
            "aggregation": "arithmetic_mean",
            "jobs": [
                {"result": {"performance": {"shapes": {"0": {"sol": {"pct": 50.0}}}}}},
                {"result": {"performance": {"shapes": {"0": {"sol": {"pct": 25.0}}}}}},
            ],
        },
        ArtifactKind.GATEWAY_RESULT,
    )

    assert gateway_result_sol_percent(artifacts, result) == 100.0 / 3.0


def test_bootstrap_staged_evaluate_uses_latency_source_stage_sol(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    result = artifacts.put_json(
        {
            "operation": "bootstrap_staged_evaluate",
            "latency_source_stage": 1,
            "completed_stages": [
                {"job": {"result": {"performance": {"shapes": {"0": {"sol": {"pct": 10.0}}}}}}},
                {"job": {"result": {"performance": {"shapes": {"0": {"sol": {"pct": 65.25}}}}}}},
            ],
        },
        ArtifactKind.GATEWAY_RESULT,
    )

    summary = gateway_result_sol_summary(artifacts, result)

    assert summary.percent == 65.25
    assert summary.source == "roofline"


def test_ordinary_comparison_aggregate_follows_raw_measurement_results(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    raw = [
        artifacts.put_json(
            {"result": {"performance": {"shapes": {"0": {"sol": {"pct": pct}}}}}},
            ArtifactKind.GATEWAY_RESULT,
        )
        for pct in (50.0, 25.0)
    ]
    aggregate = artifacts.put_json(
        {
            "operation": "evaluate_comparison",
            "aggregation": "arithmetic_mean",
            "measurements": [{"gateway_result_digest": str(digest)} for digest in raw],
        },
        ArtifactKind.GATEWAY_RESULT,
    )

    summary = gateway_result_sol_summary(artifacts, aggregate)

    assert summary.percent == 100.0 / 3.0
    assert summary.source == "roofline"


def test_same_allocation_abba_uses_authoritative_candidate_sol(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    result = artifacts.put_json(
        {
            "schema_version": 1,
            "operation": "same_allocation_abba",
            "evaluation_contract_digest": "sha256:" + "a" * 64,
            "candidate": {
                "correct": True,
                "latency_us": 12.288,
                "sol_pct": 76.19047349555397,
                "sol_pct_by_shape": {"0": 76.19047349555397},
            },
        },
        ArtifactKind.GATEWAY_RESULT,
    )

    summary = gateway_result_sol_summary(artifacts, result)

    assert summary.percent == 76.19047349555397
    assert summary.source == "roofline"
    assert summary.detail == "same-allocation ABBA Candidate aggregate"


def test_sol_percent_falls_back_to_duration_weighted_ncu_profile(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source = tmp_path / "gateway-result"
    source.mkdir()
    source.joinpath("value.json").write_text(
        '{"result":{"performance":{"shapes":{"0":{"sol":{"pct":null}}}}}}'
    )
    source.joinpath("profile.json").write_text(
        """{
          "status": "succeeded",
          "result": {
            "kernels": [
              {"compute_sol_pct": 20, "mem_sol_pct": 60, "duration": 1, "duration_unit": "us"},
              {"compute_sol_pct": 80, "mem_sol_pct": 40, "duration": 3, "duration_unit": "us"}
            ]
          }
        }"""
    )
    result = artifacts.put_directory(source, ArtifactKind.GATEWAY_RESULT)

    summary = gateway_result_sol_summary(artifacts, result)

    assert summary.percent == 75.0
    assert summary.source == "ncu-profile"
