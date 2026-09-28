"""Shared single-file and source-tree production ablation topology."""

from __future__ import annotations

from typing import Any, cast

ABLATION_OPTIMIZER_ATTEMPT_BUDGET_PER_TRAJECTORY = 15
ABLATION_PLAN_SCHEMA_VERSION = 10
ABLATION_REPLICA_COUNT = 3
ENABLED_ABLATION_ARM_KINDS = frozenset({"retained"})


def build_ablation_plan(
    policy: dict[str, Any],
    *,
    optimizer_attempt_budget_per_trajectory: int = ABLATION_OPTIMIZER_ATTEMPT_BUDGET_PER_TRAJECTORY,
) -> dict[str, Any]:
    """Derive control arms; single-file production retains its 15-Attempt default."""
    schedule = cast(dict[str, Any], policy["schedule"])
    enabled = bool(schedule.get("event_only", False))
    attempt_budget = optimizer_attempt_budget_per_trajectory
    if type(attempt_budget) is not int or attempt_budget < 1:
        raise ValueError("optimizer_attempt_budget_per_trajectory must be a positive integer")
    arms: list[dict[str, Any]] = []

    def arm(
        *,
        kind: str,
        label: str,
        attempts_per_epoch: int,
        trajectory_multiplier: int,
        workflow_command: str,
        max_challengers: int = 0,
        evolves_after_each_epoch: bool = False,
        trajectory_visibility: str = "isolated",
        tool_modules: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        if attempt_budget % 3:
            raise ValueError(
                f"Ablation Arm {label} cannot spend exactly {attempt_budget} Attempts "
                "per Trajectory: three Attempts per Epoch does not divide the budget"
            )
        target_epoch = attempt_budget // 3
        value = {
            "kind": kind,
            "label": label,
            "target_epoch_number": target_epoch,
            "workflow_command": workflow_command,
            "max_challengers": max_challengers,
            "optimizer_attempt_budget": attempts_per_epoch,
            "optimizer_attempt_budget_total": attempt_budget * trajectory_multiplier,
            "evolution_count": target_epoch if evolves_after_each_epoch else 0,
        }
        if trajectory_visibility != "isolated":
            value["trajectory_visibility"] = trajectory_visibility
        if tool_modules is not None:
            value["tool_modules"] = list(tool_modules)
        return value

    if enabled:
        # Keep the original Retained labels for the full-tool control group.
        arms.extend(
            arm(
                kind="retained",
                label=f"ablation-retained-{ordinal:02d}",
                attempts_per_epoch=3,
                trajectory_multiplier=1,
                workflow_command="workflow/retained.py",
                tool_modules=("directions", "experiments"),
            )
            for ordinal in range(1, ABLATION_REPLICA_COUNT + 1)
        )
        for suffix, modules in (
            ("no-modules", ()),
            ("experiments", ("experiments",)),
            ("directions", ("directions",)),
        ):
            arms.extend(
                arm(
                    kind="retained",
                    label=f"ablation-retained-{suffix}-{ordinal:02d}",
                    attempts_per_epoch=3,
                    trajectory_multiplier=1,
                    workflow_command="workflow/retained.py",
                    tool_modules=modules,
                )
                for ordinal in range(1, ABLATION_REPLICA_COUNT + 1)
            )
        # One retained Pool runs three parallel Trajectories, each with three serial Attempts.
        # `3` in the label denotes the number of pooled Trajectories.
        arms.append(
            arm(
                kind="pool-retained",
                label="ablation-pool-retained-3",
                attempts_per_epoch=9,
                trajectory_multiplier=3,
                workflow_command="workflow/pool_retained_3.py",
            )
        )
        arms.append(
            arm(
                kind="broadcast",
                label="ablation-broadcast-3",
                attempts_per_epoch=9,
                trajectory_multiplier=3,
                workflow_command="workflow/broadcast_3.py",
                trajectory_visibility="broadcast",
            )
        )
        arms.extend(
            arm(
                kind="retained-evolve",
                label=f"ablation-retained-evolve-{ordinal:02d}",
                attempts_per_epoch=3,
                trajectory_multiplier=1,
                workflow_command="workflow/evolve_retained_3.py",
                max_challengers=1,
                evolves_after_each_epoch=True,
            )
            for ordinal in range(1, ABLATION_REPLICA_COUNT + 1)
        )
    return {
        "schema_version": ABLATION_PLAN_SCHEMA_VERSION,
        "enabled": enabled,
        "main_evolve_enabled": False,
        "optimizer_attempt_budget_per_trajectory": attempt_budget,
        "arms": [arm for arm in arms if arm["kind"] in ENABLED_ABLATION_ARM_KINDS],
    }
