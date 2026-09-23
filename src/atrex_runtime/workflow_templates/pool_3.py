#!/usr/bin/env python3
"""Pool-3 arm: two reset-State Trajectories, each with three serial Attempts."""

from __future__ import annotations

from runtime import EpochRound, EpochRuntime, serve  # type: ignore[import-not-found]


def run_epoch(epoch: EpochRuntime) -> None:
    pool = epoch.create_pool(
        branch="active",
        trajectories=2,
        rounds=3,
    )
    def broadcast_best_kernel(current: EpochRound) -> None:
        if current.number >= pool.rounds:
            return
        best = current.best_accepted_kernel(pool)
        for outcome in current.outcomes(pool):
            current.route_kernel(
                pool,
                trajectory_ordinal=int(outcome["trajectory_ordinal"]),
                kernel_revision_id=(
                    best or str(outcome["trajectory_kernel_revision_id"])
                ),
            )

    epoch.run_pools([pool], after_round=broadcast_best_kernel)
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
