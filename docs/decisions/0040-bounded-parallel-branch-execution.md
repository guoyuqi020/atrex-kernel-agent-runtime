# Decision 0040: Bounded parallel Branch execution

## Decision

Evolver invocations remain serial until the configured Challenger pool is complete. Runtime then
runs the Active Branch and every Challenger Branch concurrently, bounded by deployment setting
`max_parallel_branches` (positive, default `4`). Within an admitted Branch, its configured
Trajectories remain concurrent and every Trajectory's Attempts remain serial.

All Branches use the same frozen Epoch starting Kernel and Evidence. They cannot consume sibling
intermediate results. Agent selection begins only after every Branch finishes successfully.

Runtime captures a Branch exception inside its task instead of letting task-group cancellation stop
sibling Branches. Siblings may finish and persist their Attempts before Runtime propagates the
deterministically selected failure. An exhausted infrastructure retry still fails the Epoch after
sibling cleanup; an unexpected process interruption preserves the running Epoch for normal resume.

## Consequences

The maximum concurrent Optimizer Session count is
`min(1 + K, max_parallel_branches) × Y`. The limit is Runtime deployment policy rather than immutable
Campaign topology, so operators can match model, Gateway, and GPU capacity without changing Lineage
identity.
