#!/usr/bin/env python3
"""Retained arm: one serial Trajectory that carries adaptive State between Attempts."""

from __future__ import annotations

from runtime import (  # type: ignore[import-not-found]
    AgentStateRef,
    EpochRound,
    EpochRuntime,
    serve,
)


def run_epoch(epoch: EpochRuntime) -> None:
    budget = int(epoch.limits["optimizer_attempts"])
    pool = epoch.create_pool(
        branch="active",
        trajectories=1,
        rounds=budget,
    )

    def carry_state(current: EpochRound) -> None:
        if current.number >= pool.rounds:
            return
        outcome = current.outcomes(pool)[0]
        current.route_kernel(
            pool,
            trajectory_ordinal=1,
            kernel_revision_id=str(outcome["trajectory_kernel_revision_id"]),
        )
        state = outcome["output_state"]
        if not isinstance(state, AgentStateRef):
            raise TypeError("Attempt outcome omitted its Agent State")
        current.route_state(pool, trajectory_ordinal=1, state=state)

    epoch.run_pools([pool], after_round=carry_state)
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
