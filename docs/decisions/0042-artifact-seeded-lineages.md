# Decision 0042: Artifact-seeded Lineage roots

## Status

Accepted.

## Decision

An active Campaign may add a Lineage from either:

- one sealed Kernel Agent Artifact digest plus one sealed Kernel Artifact digest; or
- one existing Kernel Agent revision ID plus one existing Kernel revision ID, which resolve to those sealed Artifacts.

The selected content is reused, but Registry revision identity is not. Runtime creates a new independent
`agent-v0` and `v0`, records the source revision IDs when available, and starts the new Lineage at Epoch 1.
The Agent and Kernel must match the requested DSL. They may originate in different prior Lineages, which
allows deliberate recombination of an Agent design and a Kernel starting point.

Before publication, Runtime validates the complete Agent Bundle and independently evaluates the exact
Kernel Artifact with the target Campaign's sealed Evaluation Contract and hardware target. An incorrect
Kernel does not create a Lineage. Missing Roofline data triggers the same non-authoritative SOL Profile used
by ordinary Runtime-final evaluation. `creation_key` derives stable Lineage and root revision IDs, so an
identical request recovers safely after interruption.

This is not a second Git import boundary. Standard Campaign Bootstrap remains commit-anchored through
`base_revision.commit`; Core and Evolver source is still admitted from Git only by full Commit ID. The new
operation can select only content already sealed in Runtime CAS (directly or through registered revisions).

## Interface

- CLI: `atrex-kernel-agent-runtime seed-lineage --config ... --campaign ... --spec ...`
- Admin API: `POST /v1/admin/campaigns/{campaign_id}/lineages`

The request owns the fixed DSL, optional Optimizer/Evolver models, Epoch topology, optional initial Evidence,
and one discriminated `seed` source. The response reports the new Lineage ID, `agent-v0`, `v0`, source
provenance, authoritative Gateway result, and latency.

## Consequences

Historical optimization can be branched, recombined, or reproduced without rerunning framework baseline.
The new Lineage has no parent Lineage edge: source provenance is audit metadata, while its Agent and Kernel
version histories remain independent trees. Cross-Campaign reuse is safe because the Kernel is always
re-evaluated under the destination Campaign contract.
