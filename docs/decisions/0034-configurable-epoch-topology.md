# 0034: Agent-owned Epoch Workflow

English | [中文](0034-configurable-epoch-topology.zh.md)

## Status

Accepted. This decision supersedes the earlier configuration-driven Epoch topology recorded under
the same decision number.

## Context

Branch count, evolution timing, Trajectory layout, Attempt rounds, Kernel propagation, and adaptive
State propagation are search strategy. Encoding them as Runtime configuration makes every new
strategy require controller changes and prevents the Evolver from changing how optimization work is
organized.

Runtime still needs hard, auditable resource bounds. It must also keep evaluation, isolation,
selection, promotion, persistence, and recovery outside the untrusted Agent.

## Decision

Each Lineage freezes only two Epoch resource limits:

- `max_challengers`: the maximum number of Challenger slots a Workflow may materialize;
- `optimizer_attempt_budget`: the exact number of Optimizer Attempts the Workflow must allocate.

The versioned Agent Bundle owns `workflow/main.py`. Through a narrow Workflow SDK it may replicate
or evolve Agents, create Branch pools and Trajectories, execute concurrent rounds, and explicitly
route Kernel and adaptive State outputs into later Attempts. Runtime validates capabilities and the
exact budget, then executes those requests. It does not infer a topology or provide a fallback
schedule.

The Workflow cannot change evaluation policy, Gate policy, resource limits, sandbox authority,
Registry facts, selection, or promotion. Those remain trusted Runtime responsibilities.

## Consequences

Every Campaign must contain an executable Workflow. Different ablation arms are different Workflow
programs rather than labels interpreted by Runtime. The Evolver can change search organization by
editing the Candidate Workflow, while Runtime remains small, policy-oriented, recoverable, and
auditable.
