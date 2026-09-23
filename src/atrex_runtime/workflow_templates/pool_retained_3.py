#!/usr/bin/env python3
"""Pool-Retained-3 arm: two State-retaining Trajectories with three Attempts each."""

from __future__ import annotations

from runtime import (  # type: ignore[import-not-found]
    AgentStateRef,
    EpochRound,
    EpochRuntime,
    serve,
)


def run_epoch(epoch: EpochRuntime) -> None:
    pool = epoch.create_pool(
        branch="active",
        trajectories=2,
        rounds=3,
    )

    def carry_states(current: EpochRound) -> None:
        if current.number >= pool.rounds:
            return
        best = current.best_accepted_kernel(pool)
        for outcome in current.outcomes(pool):
            ordinal = int(outcome["trajectory_ordinal"])
            current.route_kernel(
                pool,
                trajectory_ordinal=ordinal,
                kernel_revision_id=(
                    best or str(outcome["trajectory_kernel_revision_id"])
                ),
            )
            state = outcome["output_state"]
            if not isinstance(state, AgentStateRef):
                raise TypeError("Attempt outcome omitted its Agent State")
            current.route_state(
                pool,
                trajectory_ordinal=ordinal,
                state=state,
            )

    epoch.run_pools([pool], after_round=carry_states)
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
