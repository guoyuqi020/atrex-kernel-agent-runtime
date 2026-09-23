#!/usr/bin/env python3
"""Isolated-Pool-Evolve: Active and Challenger Pools with reset State."""

from __future__ import annotations

from runtime import EpochRound, EpochRuntime, serve  # type: ignore[import-not-found]


def run_epoch(epoch: EpochRuntime) -> None:
    epoch_number = int(epoch.context["epoch_number"])
    challenger = epoch.replicate_active(1) if epoch_number == 1 else epoch.evolve_agent(1)
    branch_names = (
        ("active", "challenger-1") if challenger is not None else ("active",)
    )
    branch_count = len(branch_names)
    per_branch = int(epoch.limits["optimizer_attempts"]) // branch_count
    if per_branch % 2:
        raise ValueError("Isolated-Pool-Evolve budget must divide into two-Trajectory Pools")

    pools = [
        epoch.create_pool(
            branch=branch,
            trajectories=2,
            rounds=per_branch // 2,
        )
        for branch in branch_names
    ]

    def broadcast_branch_best(current: EpochRound) -> None:
        for pool in pools:
            if current.number >= pool.rounds:
                continue
            best = current.best_accepted_kernel(pool)
            for outcome in current.outcomes(pool):
                current.route_kernel(
                    pool,
                    trajectory_ordinal=int(outcome["trajectory_ordinal"]),
                    kernel_revision_id=(
                        best or str(outcome["trajectory_kernel_revision_id"])
                    ),
                )

    epoch.run_pools(pools, after_round=broadcast_branch_best)
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
