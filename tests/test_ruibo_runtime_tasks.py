from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data/ruibo-bench/runtime-tasks"
EXPECTED_OPERATORS = {
    "bf16_mxfp4_fused_moe",
    "dual_cache_latent_bf16",
    "fmha_mxfp8_qkv",
    "fmha_nvfp4_kv",
    "fp8_scaled_grouped_gemm",
    "gdn_postconv_mtp_bf16_state_bf16",
    "mla_absorbed_fp8",
    "nvfp4_fused_moe",
    "nvfp4_gemm",
    "paged_latent_mxfp8_nvfp4",
}


def _campaign_module():
    spec = importlib.util.spec_from_file_location(
        "ppubench_campaign_ruibo", ROOT / "scripts/ppubench/campaign.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sandbox_ruibo_tasks_prepare_with_their_frozen_contracts(tmp_path: Path) -> None:
    campaign = _campaign_module()
    assert {path.name for path in TASKS.iterdir() if path.is_dir()} == EXPECTED_OPERATORS

    for operator in sorted(EXPECTED_OPERATORS):
        task_dir = TASKS / operator
        definition = json.loads((task_dir / "task.json").read_text())
        assert definition["operator"] == operator
        assert definition["curated_root"] == "data/ruibo-bench/deployed"
        assert definition["expected_valid_shapes"] == 10
        assert definition["expected_test_shapes"] == 0

        workspace = tmp_path / operator
        campaign.prepare(
            argparse.Namespace(
                root=ROOT,
                task_dir=task_dir.relative_to(ROOT),
                workspace=workspace,
                port=None,
                creation_key=f"test-{operator}",
            )
        )
        prepared = json.loads((workspace / "prepared.json").read_text())
        assert prepared["operator"] == operator
        assert prepared["valid_shapes"] == 10
        assert prepared["test_shapes"] == 0
        assert prepared["production_gate"] is True
        assert prepared["roofline_mode"] == definition["roofline_mode"]
