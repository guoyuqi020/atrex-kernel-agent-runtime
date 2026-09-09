"""Shared single-file and source-tree production ablation topology."""

from __future__ import annotations

from typing import Any, cast

ABLATION_OPTIMIZER_ATTEMPT_BUDGET_PER_TRAJECTORY = 15


def build_ablation_plan(policy: dict[str, Any]) -> dict[str, Any]:
    """Derive control arms with 15 post-Bootstrap Active Attempts per Trajectory."""
    schedule = cast(dict[str, Any], policy["schedule"])
    enabled = bool(schedule.get("event_only", False))
    trajectories = int(schedule["trajectories_per_branch"])
    challengers = int(schedule["challenger_count"])
    default_attempts = int(schedule["attempts_per_trajectory"])
    attempt_budget = ABLATION_OPTIMIZER_ATTEMPT_BUDGET_PER_TRAJECTORY
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
        trajectories_per_branch: int = 1,
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
            "challenger_count": 0,
            "challenger_start_epoch": 2,
            "first_epoch_same_agent": False,
            "optimizer_attempt_budget_total": trajectories_per_branch * attempt_budget,
            "evolution_count": 0,
        }

    if enabled:
        # Isolated arms remain independent replicas. Each one receives the full budget.
        arms.extend(
            arm(
                kind="isolated",
                label=f"ablation-isolated-{ordinal:02d}",
                attempts_per_trajectory=default_attempts,
                ephemeral_agent_state=True,
            )
            for ordinal in range(1, total + 1)
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
                    trajectories_per_branch=2,
                ),
                arm(
                    kind="pool-retained",
                    label="ablation-pool-retained-3",
                    attempts_per_trajectory=3,
                    ephemeral_agent_state=False,
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
            )
            for ordinal in range(1, total + 1)
        )
    return {
        "schema_version": 4,
        "enabled": enabled,
        "optimizer_attempt_budget_per_trajectory": attempt_budget,
        "arms": arms,
    }
