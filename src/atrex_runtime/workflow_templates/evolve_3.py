#!/usr/bin/env python3
"""Evolve-3 arm: Active and one Challenger, each with three serial Attempts."""

from __future__ import annotations

from runtime import (  # type: ignore[import-not-found]
    AgentStateRef,
    EpochRound,
    EpochRuntime,
    serve,
)


def run_epoch(epoch: EpochRuntime) -> None:
    budget = int(epoch.limits["optimizer_attempts"])
    epoch_number = int(epoch.context["epoch_number"])
    challenger = epoch.replicate_active(1) if epoch_number == 1 else epoch.evolve_agent(1)

    if challenger is None:
        pools = [
            epoch.create_pool(
                branch="active",
                trajectories=1,
                rounds=budget,
            )
        ]
    else:
        pools = [
            epoch.create_pool(
                branch=branch,
                trajectories=1,
                rounds=3,
            )
            for branch in ("active", "challenger-1")
        ]
    def carry_states(current: EpochRound) -> None:
        for pool in pools:
            if current.number >= pool.rounds:
                continue
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

    epoch.run_pools(pools, after_round=carry_states)
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
