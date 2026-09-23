#!/usr/bin/env python3
"""Retained-Evolve: run only one replicated/evolved Challenger with retained State."""

from __future__ import annotations

from runtime import (  # type: ignore[import-not-found]
    AgentStateRef,
    EpochRound,
    EpochRuntime,
    serve,
)


def run_epoch(epoch: EpochRuntime) -> None:
    epoch_number = int(epoch.context["epoch_number"])
    challenger = epoch.replicate_active(1) if epoch_number == 1 else epoch.evolve_agent(1)
    branch = "challenger-1" if challenger is not None else "active"

    pool = epoch.create_pool(
        branch=branch,
        trajectories=1,
        rounds=int(epoch.limits["optimizer_attempts"]),
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
