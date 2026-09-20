#!/usr/bin/env python3
"""Pool-Retained-3 arm: two State-retaining Trajectories with three Attempts each."""

from __future__ import annotations

from runtime import EpochRuntime, serve  # type: ignore[import-not-found]


def run_epoch(epoch: EpochRuntime) -> None:
    pool = epoch.create_pool(
        branch="active",
        trajectories=2,
        rounds=3,
        runtime_state_policy="retain_across_attempts",
    )
    epoch.run_pools([pool])
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
