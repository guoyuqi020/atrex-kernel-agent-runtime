#!/usr/bin/env python3
"""Build an atomic summary across independent Production DSL Campaigns."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DSLS = ("cuda", "triton", "cutedsl")
CAMPAIGN_ID = re.compile(r"campaign_[0-9a-f]{32}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("bootstrap", "campaign"))
    parser.add_argument("--target-epoch", type=int)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="record missing DSL results instead of failing the whole summary",
    )
    arguments = parser.parse_args()
    if arguments.phase == "campaign" and (
        arguments.target_epoch is None or arguments.target_epoch < 1
    ):
        parser.error("--target-epoch must be positive for the campaign phase")
    if arguments.phase == "bootstrap" and arguments.target_epoch is not None:
        parser.error("--target-epoch is only valid for the campaign phase")
    return arguments


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"DSL result is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"DSL result is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"DSL result must be one JSON object: {path}")
    return value


def _validate(value: dict[str, Any], *, dsl: str, phase: str, target_epoch: int | None) -> None:
    campaign_id = value.get("campaign_id")
    if not isinstance(campaign_id, str) or CAMPAIGN_ID.fullmatch(campaign_id) is None:
        raise SystemExit(f"{dsl} {phase} result has no valid campaign_id")
    lineages = value.get("lineages")
    if not isinstance(lineages, list) or len(lineages) != 1:
        raise SystemExit(f"{dsl} {phase} result must contain exactly one Lineage")
    lineage = lineages[0]
    if not isinstance(lineage, dict):
        raise SystemExit(f"{dsl} {phase} result contains an invalid Lineage")
    if phase == "campaign" and lineage.get("dsl") != dsl:
        raise SystemExit(f"{dsl} Campaign result belongs to another DSL")
    if phase == "campaign" and value.get("target_epoch_number") != target_epoch:
        raise SystemExit(f"{dsl} Campaign result has an unexpected target Epoch")


def _ablation_arms(
    dsl_workspace: Path,
    *,
    dsl: str,
    plan: dict[str, Any],
    allow_partial: bool,
) -> list[dict[str, Any]]:
    """Collect each control arm's Campaign result so the comparison pairing is durable."""
    schedules = {
        str(arm["label"]): {
            "first_epoch_same_agent": bool(arm["first_epoch_same_agent"]),
            "ephemeral_agent_state": bool(arm["ephemeral_agent_state"]),
            **{
                key: int(arm[key])
                for key in (
                    "target_epoch_number",
                    "trajectories_per_branch",
                    "attempts_per_trajectory",
                    "challenger_count",
                    "challenger_start_epoch",
                    "optimizer_attempt_budget_total",
                    "evolution_count",
                )
            },
        }
        for arm in plan.get("arms", [])
    }
    arms: list[dict[str, Any]] = []
    for arm_workspace in sorted(dsl_workspace.glob("ablation-*")):
        if not arm_workspace.is_dir():
            continue
        try:
            schedule = schedules[arm_workspace.name]
        except KeyError as error:
            raise SystemExit(
                f"{dsl} ablation arm is absent from the frozen plan: {arm_workspace.name}"
            ) from error
        path = arm_workspace / "campaign-result.json"
        seed_path = arm_workspace / "seed-result.json"
        if not path.is_file():
            if not allow_partial:
                raise SystemExit(f"{dsl} ablation arm result is missing: {path}")
            arms.append(
                {
                    "arm": arm_workspace.name,
                    "result": None,
                    "result_path": str(path),
                    "status": "missing",
                    **schedule,
                }
            )
            continue
        value = _load(path)
        _validate(value, dsl=dsl, phase="campaign", target_epoch=schedule["target_epoch_number"])
        arms.append(
            {
                "arm": arm_workspace.name,
                "campaign_id": value["campaign_id"],
                "result": value,
                "result_path": str(path),
                "seed_result": _load(seed_path) if seed_path.is_file() else None,
                **schedule,
            }
        )
    return arms


def main() -> None:
    arguments = _arguments()
    workspace = arguments.workspace.expanduser().resolve()
    ablation_plan = _load(workspace / "ablation.json") if arguments.phase == "campaign" else {}
    if arguments.phase == "campaign" and ablation_plan.get("schema_version") != 4:
        raise SystemExit("campaign summary requires Ablation Plan schema 4")
    results: dict[str, Any] = {}
    for dsl in DSLS:
        dsl_workspace = workspace / "dsls" / dsl
        campaign_path = dsl_workspace / "campaign.json"
        campaign = _load(campaign_path)
        lineages = campaign.get("lineages")
        if not isinstance(lineages, dict) or tuple(lineages) != (dsl,):
            raise SystemExit(f"{dsl} workspace does not contain exactly its own Campaign")
        path = dsl_workspace / f"{arguments.phase}-result.json"
        ablation = (
            _ablation_arms(
                dsl_workspace,
                dsl=dsl,
                plan=ablation_plan,
                allow_partial=arguments.allow_partial,
            )
            if arguments.phase == "campaign"
            else []
        )
        if arguments.allow_partial and not path.is_file():
            bootstrap_path = dsl_workspace / "bootstrap-result.json"
            results[dsl] = {
                "arm": f"evolve-{campaign['attempts_per_trajectory']}",
                "result": None,
                "result_path": str(path),
                "status": "missing",
                "bootstrap_result_path": (
                    str(bootstrap_path) if bootstrap_path.is_file() else None
                ),
                "ablation": ablation,
            }
            continue
        value = _load(path)
        _validate(
            value,
            dsl=dsl,
            phase=arguments.phase,
            target_epoch=arguments.target_epoch,
        )
        results[dsl] = {
            "arm": f"evolve-{campaign['attempts_per_trajectory']}",
            "campaign_id": value["campaign_id"],
            "result": value,
            "result_path": str(path),
            "ablation": ablation,
        }
    summary = {
        "schema_version": 1,
        "phase": arguments.phase,
        "target_epoch_number": arguments.target_epoch,
        "ablation_optimizer_attempt_budget_per_trajectory": ablation_plan.get(
            "optimizer_attempt_budget_per_trajectory"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "dsls": results,
    }
    destination = workspace / f"{arguments.phase}-results.json"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(destination)
    print(f"{arguments.phase.capitalize()} summary: {destination}")


if __name__ == "__main__":
    main()
