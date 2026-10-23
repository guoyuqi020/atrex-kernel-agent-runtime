from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ppubench_campaign", ROOT / "scripts/ppubench/campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def test_task_evaluator_version_pins_both_runtime_gate_views() -> None:
    cfg = {
        "gate_policy": {"evaluator": {}},
        "campaign": {"gate_policy": {"evaluator": {}}},
    }

    version = CAMPAIGN.pin_atrex_bench_version(
        cfg, {"atrex_bench_version": "0.1.1.dev20261020020502"}
    )

    assert version == "0.1.1.dev20261020020502"
    assert cfg["gate_policy"]["evaluator"]["agate_package_version"] == version
    assert (
        cfg["campaign"]["gate_policy"]["evaluator"]["agate_package_version"]
        == version
    )


def test_blank_task_evaluator_version_is_rejected() -> None:
    cfg = {
        "gate_policy": {"evaluator": {}},
        "campaign": {"gate_policy": {"evaluator": {}}},
    }

    with pytest.raises(ValueError, match="must be a nonempty string"):
        CAMPAIGN.pin_atrex_bench_version(cfg, {"atrex_bench_version": " "})


def test_task_evaluation_contract_payload_includes_roofline(tmp_path: Path) -> None:
    (tmp_path / "reference.py").write_text("def run():\n    return 1\n")
    (tmp_path / "input.py").write_text("def generate():\n    return {}\n")
    metadata = {"schema_version": "atrex.collection_operator_metadata.v3"}
    roofline = {
        "shapes": {
            "shape-1": {
                "semantic_W_flops": {"float8_e4m3fn": 1024},
                "semantic_Q_read_bytes": 512,
                "semantic_Q_write_bytes": 256,
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "roofline.json").write_text(json.dumps(roofline))
    shapes = {"shape-1": {"m": 1}}

    payload = CAMPAIGN.task_evaluation_contract_payload(tmp_path, shapes)

    assert payload == {
        "reference_py": "def run():\n    return 1\n",
        "input_py": "def generate():\n    return {}\n",
        "shapes": shapes,
        "metadata": metadata,
        "roofline": roofline,
    }
