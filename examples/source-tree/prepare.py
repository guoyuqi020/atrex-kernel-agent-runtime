#!/usr/bin/env python3
"""Prepare a source-tree Campaign without starting services, models, or GPU jobs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from atrex_runtime.ablation_plan import build_ablation_plan
from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.gateway.contract import AgateEvaluationContractV1
from atrex_runtime.kernel_sources import SourceManifest
from atrex_runtime.workers.problem_generalization import validate_public_operator_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-repository", required=True, type=Path)
    parser.add_argument(
        "--task",
        required=True,
        type=Path,
        help="ATREX task with reference.py, input.py, shape_train.json, shape_valid.json",
    )
    parser.add_argument("--optimizer-commit", required=True)
    parser.add_argument(
        "--hardware-target", required=True, help="Agate GPU selector, not inferred from DSL"
    )
    parser.add_argument("--dsl", choices=("cuda", "triton", "cutedsl"), default="cutedsl")
    parser.add_argument("--creation-key", default="gdn-source-tree")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; never overwrite an existing run"
    )
    args = parser.parse_args()
    manifest_path = args.source_manifest.resolve()
    manifest = SourceManifest.model_validate_json(manifest_path.read_bytes())
    task = args.task.resolve()
    shapes = json.loads((task / "shape_valid.json").read_bytes())
    public = json.loads((task / "shape_train.json").read_bytes())
    validate_public_operator_contract(public, private_shapes=shapes)
    contract = AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py=(task / "reference.py").read_text(),
        input_py=(task / "input.py").read_text(),
        shapes=shapes,
        metadata=(
            json.loads((task / "metadata.json").read_bytes())
            if (task / "metadata.json").is_file()
            else None
        ),
        roofline=(
            json.loads((task / "roofline.json").read_bytes())
            if (task / "roofline.json").is_file()
            else None
        ),
        options={
            "num_correctness_cases": 5,
            "bench_iters": 100,
            "atol": 0.01,
            "rtol": 0.05,
            "timeout_s": 600,
        },
        lock_clocks=True,
    )
    definition = {
        "schema_version": 3,
        "creation_key": args.creation_key,
        "operator": task.name,
        "hardware_target": args.hardware_target,
        "evaluation_contract": "evaluation-contract.json",
        "shape_train": "shape_train.json",
        "base_revision": {"commit": args.optimizer_commit},
        "max_challengers": 1,
        "optimizer_attempt_budget": args.attempts * 2,
        "lineages": {
            args.dsl: {
                "source_manifest": "source_manifest.json",
                "source_repository": str(args.source_repository.resolve()),
                "initial_evidence": "initial-evidence",
            }
        },
    }
    CampaignSpecV3.model_validate(definition)
    plan = build_ablation_plan(
        {"schedule": {**definition, "event_only": True}},
        optimizer_attempt_budget_per_trajectory=300,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Copy only the task declarations and fixed adapter, never a repro's history/SOTA.
    for name, value in (
        ("campaign.json", definition),
        ("ablation.json", plan),
        ("evaluation-contract.json", contract.model_dump(mode="json")),
        ("shape_train.json", public),
        ("source_manifest.json", manifest.model_dump(mode="json")),
    ):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    adapter = output / manifest.adapter
    adapter.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path.parent / manifest.adapter, adapter)
    evidence = output / "initial-evidence"
    evidence.mkdir()
    (evidence / "README.md").write_text(
        "Initial source seed; no historical optimizer evidence imported.\n"
    )
    print(f"Campaign definition: {output / 'campaign.json'}")
    print("Prepared only. No Runtime, model session, or GPU evaluation was started.")


if __name__ == "__main__":
    main()
