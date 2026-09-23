#!/usr/bin/env python3
"""Isolated-Evolve: run only one replicated/evolved Challenger with reset State."""

from __future__ import annotations

from runtime import EpochRound, EpochRuntime, serve  # type: ignore[import-not-found]


def run_epoch(epoch: EpochRuntime) -> None:
    epoch_number = int(epoch.context["epoch_number"])
    challenger = epoch.replicate_active(1) if epoch_number == 1 else epoch.evolve_agent(1)
    branch = "challenger-1" if challenger is not None else "active"

    pool = epoch.create_pool(
        branch=branch,
        trajectories=1,
        rounds=int(epoch.limits["optimizer_attempts"]),
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
