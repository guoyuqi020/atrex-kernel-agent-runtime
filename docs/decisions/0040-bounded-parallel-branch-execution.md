# Decision 0040: Bounded parallel Attempt execution

## Decision

Executable Agent Workflow code decides which Branches and Trajectories exist, which Attempts form
one logical round, and which Kernel and Agent State feed the next round. Runtime does not infer or
reconstruct that topology from Campaign parameters.

When Workflow code submits one `run_attempts_parallel` batch, Runtime admits at most
`max_parallel_attempts` Optimizer Sessions at once (positive, default `4`). Remaining launches wait
inside the same trusted operation. The limit controls deployment pressure only; it does not change
the frozen Workflow plan, ordering constraints within a Trajectory, or Lineage identity.

Runtime captures each admitted Attempt failure instead of letting task-group cancellation discard
sibling results. Siblings may finish and persist their outcomes before Runtime propagates the
deterministically selected failure. Infrastructure recovery remains Runtime policy.

## Consequences

Workflow code can express serial search, parallel pools, Active/Challenger competitions, Kernel
broadcast, and explicit Agent-State routing without adding topology switches to Runtime config.
Operators can independently bound Provider, Gateway, and GPU pressure with
`max_parallel_attempts`.
