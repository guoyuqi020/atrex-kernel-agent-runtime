# Decision 0040: Bounded parallel Branch execution

## Decision

Evolver invocations remain serial until the configured Challenger pool is complete. Runtime then
runs the Active Branch and every Challenger Branch concurrently, bounded by deployment setting
`max_parallel_branches` (positive, default `4`). Within an admitted Branch, its configured
Trajectories remain concurrent and every Trajectory's Attempts remain serial.

Controlled Challenger-only evolution workflows are the sole exception: they intentionally omit
Active and run only `challenger-1`, so there is no same-Epoch Agent comparison. The Challenger still competes
against the frozen starting Kernel for Kernel retention and is the sole eligible next Agent.

All Branches use the same frozen Epoch starting Kernel and Evidence. They cannot consume sibling
intermediate results. Agent selection begins only after every Branch finishes successfully.

Runtime captures a Branch exception inside its task instead of letting task-group cancellation stop
sibling Branches. Siblings may finish and persist their Attempts before Runtime propagates the
deterministically selected failure. An exhausted infrastructure retry still fails the Epoch after
sibling cleanup; an unexpected process interruption preserves the running Epoch for normal resume.

## Consequences

The maximum concurrent Optimizer Session count is
`min(B, max_parallel_branches) × Y`, where `B` is the number of Branches actually selected by the
validated Workflow (normally `1 + K`, and `1` for Challenger-only evolution). The limit is Runtime deployment policy rather than immutable
Campaign topology, so operators can match model, Gateway, and GPU capacity without changing Lineage
identity.
