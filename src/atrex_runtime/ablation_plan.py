"""Shared single-file and source-tree production ablation topology."""

from __future__ import annotations

from typing import Any, cast

ABLATION_OPTIMIZER_ATTEMPT_BUDGET_PER_TRAJECTORY = 15
ABLATION_PLAN_SCHEMA_VERSION = 9
ABLATION_REPLICA_COUNT = 3


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
        evolves_after_first_epoch: bool = False,
        observer_label: str | None = None,
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
            "evolution_count": max(0, target_epoch - 1) if evolves_after_first_epoch else 0,
        }
        if observer_label is not None:
            value["observer_label"] = observer_label
        return value

    if enabled:
        # The active matrix uses three independent replicas for every enabled topology.  The
        # retained-evolve, isolated-pool-evolve, and main evolve-3 Workflows remain available in
        # the Bundle/Runtime, but this plan intentionally does not schedule them.
        for kind, label, workflow in (
            ("isolated", "ablation-isolated", "workflow/isolated.py"),
            ("retained", "ablation-retained", "workflow/retained.py"),
        ):
            arms.extend(
                arm(
                    kind=kind,
                    label=f"{label}-{ordinal:02d}",
                    attempts_per_epoch=3,
                    trajectory_multiplier=1,
                    workflow_command=workflow,
                )
                for ordinal in range(1, ABLATION_REPLICA_COUNT + 1)
            )
        # Repeat each two-Trajectory Pool independently three times. The suffix after `pool-3`
        # is the replica ordinal; `3` continues to denote three serial rounds per Trajectory.
        for kind, label, workflow in (
            ("pooled", "ablation-pool-3", "workflow/pool_3.py"),
            (
                "pool-retained",
                "ablation-pool-retained-3",
                "workflow/pool_retained_3.py",
            ),
        ):
            arms.extend(
                arm(
                    kind=kind,
                    label=f"{label}-{ordinal:02d}",
                    attempts_per_epoch=6,
                    trajectory_multiplier=2,
                    workflow_command=workflow,
                )
                for ordinal in range(1, ABLATION_REPLICA_COUNT + 1)
            )
        # Run the replicated/evolved Agent alone. Each replica observes the corresponding
        # Isolated control only through its preceding Epoch; the two Lineages otherwise keep
        # independent Kernels, Agent revisions, Runtime State, Journals, and sessions.
        arms.extend(
            arm(
                kind="isolated-evolve",
                label=f"ablation-isolated-evolve-{ordinal:02d}",
                attempts_per_epoch=3,
                trajectory_multiplier=1,
                workflow_command="workflow/evolve_isolated_3.py",
                max_challengers=1,
                evolves_after_first_epoch=True,
                observer_label=f"ablation-isolated-{ordinal:02d}",
            )
            for ordinal in range(1, ABLATION_REPLICA_COUNT + 1)
        )
    return {
        "schema_version": ABLATION_PLAN_SCHEMA_VERSION,
        "enabled": enabled,
        "main_evolve_enabled": False,
        "optimizer_attempt_budget_per_trajectory": attempt_budget,
        "arms": arms,
    }
