"""Shared single-file and source-tree production ablation topology."""

from __future__ import annotations

from typing import Any, cast

ABLATION_OPTIMIZER_ATTEMPT_BUDGET_PER_TRAJECTORY = 15
ABLATION_PLAN_SCHEMA_VERSION = 5


def build_ablation_plan(
    policy: dict[str, Any],
    *,
    optimizer_attempt_budget_per_trajectory: int = ABLATION_OPTIMIZER_ATTEMPT_BUDGET_PER_TRAJECTORY,
) -> dict[str, Any]:
    """Derive control arms; single-file production retains its 15-Attempt default."""
    schedule = cast(dict[str, Any], policy["schedule"])
    enabled = bool(schedule.get("event_only", False))
    trajectories = int(schedule["trajectories_per_branch"])
    challengers = int(schedule["challenger_count"])
    default_attempts = int(schedule["attempts_per_trajectory"])
    attempt_budget = optimizer_attempt_budget_per_trajectory
    if type(attempt_budget) is not int or attempt_budget < 1:
        raise ValueError("optimizer_attempt_budget_per_trajectory must be a positive integer")
    # Pair Isolated and Retained replicas for every configured Active/Challenger Trajectory. The
    # configured Challenger count is used rather than the challenger_start_epoch-gated one
    # so arm identity is stable across Epochs.
    total = trajectories * (1 + challengers)
    arms: list[dict[str, Any]] = []

    def arm(
        *,
        kind: str,
        label: str,
        attempts_per_trajectory: int,
        ephemeral_agent_state: bool,
        workflow_command: str,
        trajectories_per_branch: int = 1,
        challenger_count: int = 0,
        challenger_start_epoch: int = 2,
        first_epoch_same_agent: bool = False,
        runs_active_branch: bool = True,
    ) -> dict[str, Any]:
        if attempt_budget % attempts_per_trajectory:
            raise ValueError(
                f"Ablation Arm {label} cannot spend exactly {attempt_budget} Attempts "
                "per Trajectory: "
                f"{attempts_per_trajectory} Attempts per Epoch does not divide the budget"
            )
        target_epoch = attempt_budget // attempts_per_trajectory
        return {
            "kind": kind,
            "label": label,
            "trajectories_per_branch": trajectories_per_branch,
            "attempts_per_trajectory": attempts_per_trajectory,
            "target_epoch_number": target_epoch,
            "ephemeral_agent_state": ephemeral_agent_state,
            "workflow_command": workflow_command,
            "challenger_count": challenger_count,
            "challenger_start_epoch": challenger_start_epoch,
            "first_epoch_same_agent": first_epoch_same_agent,
            "optimizer_attempt_budget_total": (
                trajectories_per_branch
                * attempt_budget
                * (int(runs_active_branch) + challenger_count)
            ),
            "evolution_count": (
                max(0, target_epoch - challenger_start_epoch + 1) * challenger_count
            ),
        }

    if enabled:
        # Isolated arms remain independent replicas. Each one receives the full budget.
        arms.extend(
            arm(
                kind="isolated",
                label=f"ablation-isolated-{ordinal:02d}",
                attempts_per_trajectory=default_attempts,
                ephemeral_agent_state=True,
                workflow_command="workflow/isolated.py",
            )
            for ordinal in range(1, total + 1)
        )
        # Run the replicated/evolved Agent alone. There is no same-Epoch Active comparator,
        # and State resets before every Attempt.
        arms.extend(
            arm(
                kind="isolated-evolve",
                label=f"ablation-isolated-evolve-{ordinal:02d}",
                attempts_per_trajectory=3,
                ephemeral_agent_state=True,
                workflow_command="workflow/evolve_isolated_3.py",
                challenger_count=1,
                first_epoch_same_agent=True,
                runs_active_branch=False,
            )
            for ordinal in range(1, 3)
        )
        # Keep the same Challenger-only organization while retaining adaptive State across the
        # three serial Attempts. Paired replicas separate the State-retention effect from noise.
        arms.extend(
            arm(
                kind="retained-evolve",
                label=f"ablation-retained-evolve-{ordinal:02d}",
                attempts_per_trajectory=3,
                ephemeral_agent_state=False,
                workflow_command="workflow/evolve_retained_3.py",
                challenger_count=1,
                first_epoch_same_agent=True,
                runs_active_branch=False,
            )
            for ordinal in range(1, 3)
        )
        # Combine reset-State Pool search with Agent evolution. Active and Challenger each run two
        # independent Trajectories; Optimizer-produced adaptive State never survives an Attempt.
        arms.append(
            arm(
                kind="isolated-pool-evolve",
                label="ablation-isolated-pool-evolve-3",
                attempts_per_trajectory=3,
                ephemeral_agent_state=True,
                workflow_command="workflow/evolve_isolated_pool_3.py",
                trajectories_per_branch=2,
                challenger_count=1,
                first_epoch_same_agent=True,
            )
        )
        # Pair two-Trajectory Pools at three Attempts per Epoch, differing only in
        # whether adaptive Runtime State survives each Attempt. Each has its full budget.
        arms.extend(
            (
                arm(
                    kind="pooled",
                    label="ablation-pool-3",
                    attempts_per_trajectory=3,
                    ephemeral_agent_state=True,
                    workflow_command="workflow/pool_3.py",
                    trajectories_per_branch=2,
                ),
                arm(
                    kind="pool-retained",
                    label="ablation-pool-retained-3",
                    attempts_per_trajectory=3,
                    ephemeral_agent_state=False,
                    workflow_command="workflow/pool_retained_3.py",
                    trajectories_per_branch=2,
                ),
            )
        )
        # Match each Isolated replica, retaining only its own adaptive Runtime State.
        arms.extend(
            arm(
                kind="retained",
                label=f"ablation-retained-{ordinal:02d}",
                attempts_per_trajectory=default_attempts,
                ephemeral_agent_state=False,
                workflow_command="workflow/retained.py",
            )
            for ordinal in range(1, total + 1)
        )
    return {
        "schema_version": ABLATION_PLAN_SCHEMA_VERSION,
        "enabled": enabled,
        "optimizer_attempt_budget_per_trajectory": attempt_budget,
        "arms": arms,
    }
