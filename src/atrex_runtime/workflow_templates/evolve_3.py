#!/usr/bin/env python3
"""Evolve-3 arm: Active and one Challenger, each with three serial Attempts."""

from __future__ import annotations

from runtime import EpochRuntime, serve  # type: ignore[import-not-found]


def run_epoch(epoch: EpochRuntime) -> None:
    budget = int(epoch.limits["optimizer_attempts"])
    epoch_number = int(epoch.context["epoch_number"])
    first_epoch_same_agent = bool(epoch.context["first_epoch_same_agent"])
    challenger: str | None
    if epoch_number == 1 and first_epoch_same_agent:
        challenger = epoch.replicate_active(1)
    else:
        challenger = epoch.evolve_agent(1)

    if challenger is None:
        pools = [
            epoch.create_pool(
                branch="active",
                trajectories=1,
                rounds=budget,
                runtime_state_policy="retain_across_attempts",
            )
        ]
    else:
        pools = [
            epoch.create_pool(
                branch=branch,
                trajectories=1,
                rounds=3,
                runtime_state_policy="retain_across_attempts",
            )
            for branch in ("active", "challenger-1")
        ]
    epoch.run_pools(pools)
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
