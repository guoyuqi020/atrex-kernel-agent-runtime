# Decision 0033: Version evolved Kernel Agent revisions independently from Kernels

English | [中文](0033-lineage-agent-version-labels.zh.md)

## Status

Accepted and implemented.

## Context

Kernel Agent revisions already have opaque IDs, parent links, creation times, and sealed Optimizer
Artifact Digests. Those fields provide exact identity but do not make a Lineage's evolving Harness
history readable. Kernel `vN` labels cannot be reused: one Agent revision can produce many Kernels,
and a rejected Challenger can be followed by a sibling evolved from the same Active parent.

## Decision

Registry schema 17 adds `lineage_agent_versions`. It assigns each Agent revision one immutable,
zero-based number within exactly one Lineage. Bootstrap links the initial Optimizer as `agent-v0`.
An Evolver output receives the next number only when the trusted controller attaches the validated
Challenger to its Epoch. Attachment verifies that the Challenger parent is the Epoch's Active Agent;
mapping and attachment commit in one transaction.

The mapping stores introduction Epoch and link time. Migration from schema 16 reconstructs revisions
in Epoch order. Catalog projections include Agent/parent versions, creation source and time,
introduction Epoch, active flag, disposition (`baseline`, `challenger`, `promoted`, `rejected`, or
`failed`), exact IDs, Optimizer Artifact, and trace/provenance Digests. Kernel projections include
the producing `kernel_agent_version`.

Authenticated API routes and the `list-agent-revisions`/`show-agent-revision` CLI expose this
history. JSON remains the automation default; `--format table` is the operator view.

## Consequences

Kernel and Agent counters are independent. A single `agent-v0` may produce `v0`, `v1`, and `v2`
Kernels. Agent numeric adjacency is not ancestry: `agent-v2` may have `parent_agent_version` equal to
`agent-v0` when `agent-v1` lost promotion. Git Commit and Artifact Digest remain the supply-chain
and executed-content identities; the Lineage label is only a stable human projection.
