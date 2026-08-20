# ADR 0036: Frozen Runtime Tools for Evolver

English | [中文](0036-frozen-evolver-runtime-tools.zh.md)

## Status

Accepted and implemented.

## Context

ADR 0035 exposes all completed branches and exact Kernel artifacts to Evolver, but manual traversal
of a long Epoch tree is expensive and error-prone. Runtime administration APIs can already render
versioned histories, but granting their bearer token to Evolver would expose live mutable state and
authority far beyond one Evolution session. A repository-owned helper would itself be evolvable or
could drift from Runtime's Evidence contract.

## Decision

Evolution Input schema v4 adds the fixed `runtime-tools/` path. Before each Evolver process starts,
Runtime freezes:

- `catalog.json`, containing exact Lineage-local `vN` and `agent-vN` labels, revision identities,
  parent links, provenance, evaluation facts, dispositions, and source paths;
- `kernels/<kernel-revision-id>/`, containing every exact historical Lineage Kernel Artifact; and
- `evolver_tools.py`, a Runtime-owned, standard-library-only inspection and constrained
  Candidate-control client.

The client exposes bounded JSON `history`, `branches`, `attempts`, `kernels`, `kernel-read`, `agents`,
`agent-diff`, and `trace-paths` commands. Its sole mutation is
`candidate-reset --base <agentrev>`: it accepts only an immutable Manifest entry marked
`lineage_history`, rejects links and special files, stages a complete writable copy, atomically
replaces only `candidate/`, and records the chosen base in `scratch/candidate-base.json`. Runtime
seals the tool and all input directories read-only before launch. Evolver's Session context supplies
the exact interpreter/command pair; final sealing reconciles the base record, proposal, and actual
repository diff.

This is a local frozen-workspace surface, not an HTTP capability. It carries no Admin, Registry,
Gateway, Wiki, evaluation, or promotion credential and cannot observe changes made after workspace
preparation. Direct Evidence and repository files remain available for verification.

## Consequences

- Evolver gets a deterministic inspect workflow without weakening the trusted control boundary.
- Version labels come from Registry catalogs rather than being inferred from directory order.
- All historical Kernel sources are available even before the first completed Epoch.
- Evolution workspaces grow with Lineage Kernel history; retention sizing remains an operational
  concern.
- Older Evolver Bundles that validate schema v3 cannot consume schema v4 and require an explicit
  pinned-Commit upgrade.
