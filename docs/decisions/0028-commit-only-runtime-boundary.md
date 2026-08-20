# Decision 0028: Commit-only Agent bundles and a minimal Runtime launcher

English | [中文](0028-commit-only-runtime-boundary.zh.md)

## Status

Accepted, 2026-08-18. This is the current Runtime/Agent source boundary.

## Decision

Runtime accepts only Campaign schema v3 and a full Core Git commit. The Campaign definition's `lineages` keys are its complete DSL set; deployment configuration does not duplicate that topology. Core and Evolver are complete, separately versioned repositories imported through one shared safe Git boundary and sealed as immutable Artifacts. Core and Evolver own their Agent framework, prompt, and Backend adapters; Runtime deployment owns the Backend while Campaign/Lineage state owns concrete model selection. Runtime records Evolver commit, tree, Artifact, and launch fingerprints and sends one fixed stdin instruction.

Runtime exposes one launcher contract with explicit `development` and Linux `sandbox` modes. The
production mode applies bubblewrap and cgroup v2 to the complete
Core/Evolver process tree; Runtime never treats the unisolated development mode as a fallback.

Evidence persists normalized projections plus Session source digests, not duplicate raw bytes. Agent views materialize original unredacted Session Artifacts by digest. Wiki feedback constructs its own exact bounded raw projection after epoch completion.

## Consequences

Configuration and provenance have one source of truth and dead compatibility paths are removed.
Whole-Worker isolation is defined by the current sandbox and network-namespace decisions; no
framework-specific historical design is evidence of current behavior.
