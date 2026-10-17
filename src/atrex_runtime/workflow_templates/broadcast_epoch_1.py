#!/usr/bin/env python3
"""Three concurrent Trajectories that publish a shared checkpoint every round."""

from __future__ import annotations

from runtime import EpochRuntime, serve  # type: ignore[import-not-found]


def run_epoch(epoch: EpochRuntime) -> None:
    pool = epoch.create_pool(branch="active", trajectories=3, rounds=1)
    epoch.run_pools([pool])
    epoch.complete()


if __name__ == "__main__":
    raise SystemExit(serve(run_epoch))
