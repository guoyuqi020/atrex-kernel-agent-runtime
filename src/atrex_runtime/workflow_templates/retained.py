#!/usr/bin/env python3
"""Retained arm: one serial Trajectory that carries adaptive State between Attempts."""

from __future__ import annotations

from runtime import EpochRuntime, serve  # type: ignore[import-not-found]


def run_epoch(epoch: EpochRuntime) -> None:
    budget = int(epoch.limits["optimizer_attempts"])
    pool = epoch.create_pool(
        branch="active",
        trajectories=1,
        rounds=budget,
        runtime_state_policy="retain_across_attempts",
    )
    epoch.run_pools([pool])
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
