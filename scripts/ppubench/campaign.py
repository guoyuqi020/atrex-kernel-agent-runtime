#!/usr/bin/env python3
"""Prepare a PPU task snapshot, or run its Bootstrap followed by all ablation arms."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

DEFAULT_TASK = Path("data/ppubench/gated-residual-combine-cuda")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def git(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def prepare(args):
    from atrex_runtime.bootstrap import CampaignSpecV3
    from atrex_runtime.config import RuntimeSettings
    from atrex_runtime.gateway.contract import AgateEvaluationContractV1
    from atrex_runtime.workers.problem_generalization import validate_public_operator_contract

    root, workspace = args.root.resolve(), args.workspace.resolve()
    kit = (root / args.task_dir).resolve()
    if workspace.is_relative_to(root / "data") or workspace.is_relative_to(kit):
        raise SystemExit("Workspace must be outside task definitions in data/")
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit("Refusing nonempty workspace; choose a new workspace")

    # Freeze the complete task kit before resolving machine-specific paths and Git revisions.
    frozen = workspace / "task-definition"
    shutil.copytree(kit, frozen)
    definition, policy = read(frozen / "task.json"), read(frozen / "policy.json")
    campaign = read(frozen / "campaign.template.json")
    plan = read(frozen / "ablation.json")
    dsl, operator = definition["dsl"], definition["operator"]
    if campaign["operator"] != operator or list(campaign["lineages"]) != [dsl]:
        raise ValueError("Task and Campaign operator/DSL do not match")
    if policy["gate_policy"]["production_gate"] is not True:
        raise ValueError("PPU campaign requires the production gate")

    data = root / definition["curated_root"]
    op = data / "collections" / definition["collection"] / "operators" / operator
    task = workspace / "task"
    task.mkdir()
    for name in (
        "input.py",
        "metadata.json",
        "shape_cases.json",
        "shape_range.json",
        "roofline.json",
    ):
        shutil.copyfile(op / name, task / name)
    shutil.copyfile(data / "operators" / operator / "reference.py", task / "reference.py")
    # Deliberately do not import the curated solution.py as an optimization seed.
    shutil.copytree(frozen / "inputs", workspace / "inputs")
    shapes = read(task / "shape_cases.json")
    heads = {
        name: git(root / path)
        for name, path in (
            ("runtime", "."),
            ("core", "src/atrex-kernel-agent-core"),
            ("kda", "src/kernel-design-agents"),
            ("evolver", "src/atrex-kernel-agent-evolver"),
            ("bench", "third_party/atrex-bench"),
        )
    }
    runtime = definition["runtime"]
    helper = module("production_prepare", root / "scripts/production/prepare.py")
    cfg = helper._runtime_config(
        root=root,
        workspace=workspace,
        backend=runtime["backend"],
        hardware_target=campaign["hardware_target"],
        policy=policy,
        host=policy["runtime"]["host"],
        port=args.port if args.port is not None else policy["runtime"]["port"],
        wiki_url=policy["runtime"]["wiki_url"],
        worker_user=runtime["worker_user"],
        host_home=runtime["host_home"],
        launcher_mode=runtime["launcher_mode"],
        evolver_commit=heads["evolver"],
        bench_commit=heads["bench"],
    )
    cfg["agate"].update(runtime["agate"])
    cfg["gpu_wiki"] = runtime["gpu_wiki"]
    cfg["campaign"]["roofline_builder"] = runtime["roofline_builder"]
    cfg["campaign"]["launcher"]["backend_credentials"]["host_home"] = None
    RuntimeSettings.model_validate(cfg)
    write(workspace / "runtime.json", cfg)

    contract = read(frozen / "evaluation.template.json")
    contract.update(
        reference_py=(task / "reference.py").read_text(),
        input_py=(task / "input.py").read_text(),
        shapes=shapes,
        metadata=read(task / "metadata.json"),
    )
    parsed = AgateEvaluationContractV1.model_validate(contract).with_shape_holdout()
    if (
        len(parsed.validation_shape_ids) != definition["expected_valid_shapes"]
        or len(parsed.shape_split.test_shape_ids) != definition["expected_test_shapes"]
    ):
        raise ValueError("Evaluation shape split differs from the task definition")
    write(workspace / "evaluation-contract.json", parsed.model_dump(mode="json"))
    problem = validate_public_operator_contract(
        read(frozen / "agent-problem.json"), private_shapes=shapes
    )
    write(workspace / "agent-problem.json", problem)

    campaign.update(
        creation_key=args.creation_key or workspace.name,
        base_revision={"commit": heads["kda"]},
    )
    write(workspace / "campaign.json", campaign)
    CampaignSpecV3.from_file(workspace / "campaign.json")
    write(workspace / "ablation.json", plan)
    files = {
        str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (frozen, task)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    write(
        workspace / "prepared.json",
        {
            "operator": operator,
            "hardware_target": campaign["hardware_target"],
            "dsl": dsl,
            "fresh_bootstrap": definition["fresh_bootstrap"],
            "seed_kind": definition["seed_kind"],
            "task_definition": str(kit),
            "task_snapshot": str(frozen),
            "input_sha256": files,
            "heads": heads,
            "valid_shapes": len(parsed.validation_shape_ids),
            "test_shapes": len(parsed.shape_split.test_shape_ids),
            "arms": len(plan["arms"]),
            "trajectories_per_arm": 3,
            "target_epochs": plan["arms"][0]["target_epoch_number"],
            "attempts_per_trajectory": plan["optimizer_attempt_budget_per_trajectory"],
            "max_parallel_attempts_per_arm": cfg["campaign"]["max_parallel_attempts"],
            "bootstrap_timeout_seconds": policy["workers"]["bootstrap_timeout_seconds"],
            "model": campaign["lineages"][dsl]["models"]["optimizer"],
            "production_gate": cfg["gate_policy"]["production_gate"],
            "lock_clocks": cfg["gate_policy"]["lock_clocks"],
            "roofline_mode": definition["roofline_mode"],
        },
    )
    print(json.dumps({"prepared": str(workspace), "heads": heads}, indent=2))


def run(args):
    from atrex_runtime.ablation import AblationArmSpecV1
    from atrex_runtime.bootstrap import CampaignSpecV3

    root, workspace = args.root.resolve(), args.workspace.resolve()
    cli = root / ".venv/bin/atrex-kernel-agent-runtime"
    helper = module("source_tree_ablation", root / "scripts/source-tree/run.py")
    spec = CampaignSpecV3.from_file(workspace / "campaign.json")
    definition = read(workspace / "task-definition/task.json")
    dsl = definition["dsl"]
    if [item.value for item in spec.lineages] != [dsl]:
        raise ValueError("Prepared Campaign DSL differs from the frozen task")
    runroot = workspace / "ablation-run"
    runroot.mkdir(exist_ok=True)
    with (runroot / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        boot = helper.run_json(
            str(cli),
            [
                "bootstrap",
                "--config",
                str(workspace / "runtime.json"),
                "--campaign",
                str(workspace / "campaign.json"),
            ],
            runroot / "bootstrap-result.json",
            runroot / "bootstrap.log",
        )
        if len(boot["lineages"]) != 1:
            raise ValueError("Expected one completed Bootstrap lineage")
        source = boot["lineages"][0]["lineage_id"]
        arms = []
        for arm in read(workspace / "ablation.json")["arms"]:
            dest = runroot / arm["label"]
            dest.mkdir(exist_ok=True)
            seed = AblationArmSpecV1(
                creation_key=f"{spec.creation_key}-{arm['label']}-{dsl}",
                source_lineage_id=source,
                trajectory_visibility=arm["trajectory_visibility"],
                tool_modules=arm["tool_modules"],
                **{
                    key: arm[key]
                    for key in ("optimizer_attempt_budget", "max_challengers", "workflow_command")
                },
            )
            write(dest / "seed.json", seed.model_dump(mode="json"))
            result = helper.run_json(
                str(cli),
                [
                    "seed-ablation-arm",
                    "--config",
                    str(workspace / "runtime.json"),
                    "--spec",
                    str(dest / "seed.json"),
                ],
                dest / "seed-result.json",
                dest / "seed.log",
            )
            arms.append(
                {
                    **arm,
                    "campaign_id": result["campaign_id"],
                    "lineage_id": result["lineage"]["lineage_id"],
                }
            )
        helper.run_arms(str(cli), workspace / "runtime.json", runroot, arms)


def terminate(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--creation-key")
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, terminate)
    (prepare if args.action == "prepare" else run)(args)
