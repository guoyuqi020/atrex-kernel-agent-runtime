#!/usr/bin/env python3
"""Isolated arm: one Agent Trajectory whose adaptive State resets every Attempt."""

from __future__ import annotations

from runtime import EpochRound, EpochRuntime, serve  # type: ignore[import-not-found]


def run_epoch(epoch: EpochRuntime) -> None:
    budget = int(epoch.limits["optimizer_attempts"])
    pool = epoch.create_pool(
        branch="active",
        trajectories=1,
        rounds=budget,
    )
    def carry_kernel(current: EpochRound) -> None:
        if current.number >= pool.rounds:
            return
        outcome = current.outcomes(pool)[0]
        current.route_kernel(
            pool,
            trajectory_ordinal=1,
            kernel_revision_id=str(outcome["trajectory_kernel_revision_id"]),
        )

    epoch.run_pools([pool], after_round=carry_kernel)
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
