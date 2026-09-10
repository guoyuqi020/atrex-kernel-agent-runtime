#!/usr/bin/env python3
"""Bootstrap once, then run the production ablation arms against a source-tree task.

Uses an existing Runtime deployment; never starts or stops shared services.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from atrex_runtime.ablation import AblationArmSpecV1
from atrex_runtime.ablation_plan import build_ablation_plan
from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.domain.ids import parse_campaign_id, parse_lineage_id


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def announce(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def stop_process(process: subprocess.Popen) -> None:
    """Reap only our own process group, including an interrupted Bootstrap."""
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_json(cli: str, arguments: list[str], output: Path, log: Path) -> dict[str, Any]:
    temporary = output.with_suffix(".tmp")
    announce(f"{arguments[0]}; log: {log}")
    with temporary.open("w") as stdout, log.open("a") as stderr:
        command = [cli, *arguments]
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, start_new_session=True,
        )
        try:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            stop_process(process)
    value = json.loads(temporary.read_text())
    parse_campaign_id(value["campaign_id"])
    temporary.replace(output)
    return value


def run_arms(cli: str, config: Path, workspace: Path, arms: list[dict[str, Any]]) -> None:
    """Independent processes: one failed arm does not cancel the others."""
    summary = {"schema_version": 1, "arms": arms}
    pending: dict[subprocess.Popen, tuple[dict[str, Any], Path]] = {}
    with ExitStack() as streams:
        try:
            for arm in arms:
                root = workspace / arm["label"]
                root.mkdir(exist_ok=True)
                output = root / "campaign-result.tmp"
                log = root / "campaign.log"
                process = subprocess.Popen(
                    [cli, "run-campaign", "--config", str(config), "--campaign",
                     arm["campaign_id"], "--target-epoch", str(arm["target_epoch_number"])],
                    stdout=streams.enter_context(output.open("w")),
                    stderr=streams.enter_context(log.open("a")),
                    start_new_session=True,
                )
                arm.update(status="running", log=str(log), result_path=None)
                pending[process] = (arm, output)
                announce(f"{arm['label']} started: {arm['campaign_id']}; log: {log}")
            write_json(workspace / "campaign-results.json", summary)
            while pending:
                for process, (arm, output) in list(pending.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    arm.update(exit_code=code, status="failed")
                    if code == 0:
                        try:
                            result = json.loads(output.read_text())
                            if (result["campaign_id"] != arm["campaign_id"] or
                                    result["target_epoch_number"] != arm["target_epoch_number"]):
                                raise ValueError("unexpected Campaign result identity/target")
                            destination = output.with_suffix(".json")
                            output.replace(destination)
                            arm.update(status="completed", result_path=str(destination))
                        except (ValueError, KeyError, TypeError) as error:
                            arm["error"] = str(error)
                    announce(f"{arm['label']} {arm['status']}; exit={code}")
                    del pending[process]
                    write_json(workspace / "campaign-results.json", summary)
                if pending:
                    time.sleep(0.2)
        finally:
            # Only terminate children owned by this invocation; shared services are untouched.
            for process in pending:
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
            for process, (arm, _) in pending.items():
                stop_process(process)
                arm["status"] = "interrupted"
            write_json(workspace / "campaign-results.json", summary)
    if any(arm["status"] != "completed" for arm in arms):
        raise SystemExit(f"Some arms failed; inspect {workspace / 'campaign-results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--target-epoch", type=int, default=100, help="main arm only")
    args = parser.parse_args()
    if args.target_epoch < 1:
        parser.error("--target-epoch must be positive")
    config, campaign_path = args.config.resolve(), args.campaign.resolve()
    if not config.is_file():
        parser.error(f"Runtime config not found: {config}")
    spec = CampaignSpecV3.from_file(campaign_path)
    if len(spec.lineages) != 1 or any(
        lineage.source_manifest is None for lineage in spec.lineages.values()
    ):
        parser.error("expected one source-tree DSL Lineage")
    if not spec.first_epoch_same_agent:
        parser.error("production ablation requires first_epoch_same_agent=true; use a new Campaign")
    definition = spec.model_dump(mode="json")
    plan = json.loads(args.plan.read_text())
    try:
        expected = build_ablation_plan(
            {"schedule": {**definition, "event_only": True}},
            optimizer_attempt_budget_per_trajectory=plan.get("optimizer_attempt_budget_per_trajectory"),
        )
    except ValueError as error:
        parser.error(str(error))
    if plan != expected:
        parser.error("Ablation Plan differs from the shared production topology/budget")
    cli_path = Path(sys.executable).absolute().parent / "atrex-kernel-agent-runtime"
    cli = str(cli_path) if cli_path.is_file() else shutil.which("atrex-kernel-agent-runtime")
    if not cli:
        parser.error("activate the Runtime Python environment first")
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".launch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("this ablation workspace already has a runner")
        frozen = {"config": str(config), "campaign": definition, "ablation": plan}
        launch_path = workspace / "launch-inputs.json"
        if launch_path.exists() and json.loads(launch_path.read_text()) != frozen:
            parser.error("frozen Campaign/Plan changed; use a new workspace and creation_key")
        write_json(launch_path, frozen)
        write_json(workspace / "ablation.json", plan)
        bootstrap = run_json(
            cli, ["bootstrap", "--config", str(config), "--campaign", str(campaign_path)],
            workspace / "bootstrap-result.json", workspace / "bootstrap.log",
        )
        if len(bootstrap["lineages"]) != 1:
            raise ValueError("Bootstrap returned more than one Lineage")
        source_id = parse_lineage_id(bootstrap["lineages"][0]["lineage_id"])
        arms = [{
            "label": f"evolve-{spec.attempts_per_trajectory}", "kind": "evolve",
            "campaign_id": bootstrap["campaign_id"], "lineage_id": str(source_id),
            "target_epoch_number": args.target_epoch,
            "optimizer_attempt_budget_total": (
                args.target_epoch * spec.attempts_per_trajectory * spec.trajectories_per_branch * 2
            ),
        }]
        for arm in plan["arms"]:
            root = workspace / arm["label"]
            root.mkdir(exist_ok=True)
            seed_spec = AblationArmSpecV1(
                creation_key=f"{arm['label']}-{next(iter(spec.lineages)).value}",
                source_lineage_id=source_id,
                **{key: arm[key] for key in (
                    "attempts_per_trajectory", "trajectories_per_branch", "ephemeral_agent_state",
                    "challenger_count", "challenger_start_epoch", "first_epoch_same_agent",
                )},
            )
            write_json(root / "seed.json", seed_spec.model_dump(mode="json"))
            seeded = run_json(
                cli, ["seed-ablation-arm", "--config", str(config),
                      "--spec", str(root / "seed.json")],
                root / "seed-result.json", root / "seed.log",
            )
            arms.append({**arm, "campaign_id": seeded["campaign_id"],
                         "lineage_id": seeded["lineage"]["lineage_id"]})
        run_arms(cli, config, workspace, arms)


if __name__ == "__main__":
    # Turn SIGTERM into orderly cleanup of this runner's child Campaign processes.
    def stop(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    main()
