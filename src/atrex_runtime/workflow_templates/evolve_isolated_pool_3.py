#!/usr/bin/env python3
"""Isolated-Pool-Evolve: Active and Challenger Pools with reset State."""

from __future__ import annotations

from runtime import EpochRuntime, serve  # type: ignore[import-not-found]


def run_epoch(epoch: EpochRuntime) -> None:
    epoch_number = int(epoch.context["epoch_number"])
    first_epoch_same_agent = bool(epoch.context["first_epoch_same_agent"])
    if epoch_number == 1 and first_epoch_same_agent:
        challenger = epoch.replicate_active(1)
    else:
        challenger = epoch.evolve_agent(1)
    if challenger is None:
        raise RuntimeError("Isolated-Pool-Evolve requires one materialized Challenger Agent")

    pools = [
        epoch.create_pool(
            branch=branch,
            trajectories=2,
            rounds=int(epoch.limits["default_attempts_per_trajectory"]),
            runtime_state_policy="reset_each_attempt",
        )
        for branch in ("active", "challenger-1")
    ]
    epoch.run_pools(pools)
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
