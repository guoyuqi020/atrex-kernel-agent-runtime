# Implementation Status

English | [中文](implementation-status.zh.md)

This document is authoritative when historical decisions or target-design text differ from current code.

| Area | Status | Current implementation | Remaining work |
| --- | --- | --- | --- |
| Campaign/Epoch control | Implemented | Durable SQLite state, idempotent transitions, renewable generation fencing, serial Challenger construction followed by bounded concurrent Active/Challenger Branches, concurrent per-Branch Trajectories, serial per-Trajectory Attempts, sibling-failure isolation, durable-completion CLI progress, recovery, cancellation/finalization | Multi-node Registry is deferred |
| Campaign bootstrap | Implemented | Separate Campaign schema v3 definition with `lineages`-owned DSL topology and model selection, optional Campaign problem-generalization model, Campaign-frozen full Evolver Commit with resume drift rejection, commit-pinned trusted Atrex Bench Roofline build with exact validation and sealed recovery, full Core commit, shared Agent Problem, mandatory Core baseline per DSL, append-only retry Generations with Session/Token/Report/Outcome audit | Real provider/Gateway acceptance and representative production Roofline cost-model coverage |
| Artifact-seeded Lineage | Implemented | Admin API/CLI accepts sealed Agent+Kernel Artifacts or registered Revision IDs, validates same-DSL Bundle content, performs destination-Campaign authoritative Agate evaluation, creates independent `agent-v0`/`v0`, preserves source provenance, and recovers idempotently | Real cross-Campaign GPU acceptance and crash-window rehearsal |
| Core Optimizer | Implemented | One Core-owned entrypoint; Runtime-bound Claude/Codex/QoderCLI/Pi selection shared by problem, baseline, and Attempt phases; live unsealed Session projection followed by terminal rebuild and sealing; live Claude/Codex/QoderCLI Sandbox connectivity verified in Lima | Full workflow acceptance per Backend and Pi connectivity |
| Evolver | Implemented | Independent Runtime-bound Claude/Codex/QoderCLI/Pi selection; lazy credential/Bundle resolution; one tagged `evolved`, `reuse`, or `evolve_from_history` Challenger proposal per invocation; frozen Agent/Kernel inspection plus atomic historical Candidate reset; commit/tree/Artifact identity; Candidate-base/Diff, same-DSL, and frozen-history validation; per-Epoch proposal provenance; failure evidence | Recursive Evolver self-evolution deferred |
| Kernel/Agent selection | Implemented | Ordinary Evaluate or commit-pinned same-allocation ABBA; durable per-run evidence | Real GPU repeatability study |
| Production Gate | Implemented | Campaign-frozen optional content gate before Agate and authoritative publication; fixed-DSL/self-contained markers, Python AST fallback checks, dynamic/prebuilt dependency rejection, and `solution.json` language/dependency validation | Curated independent reviewer for ambiguous third-party implementation provenance |
| Evidence | Implemented | Branch-local Optimizer Evidence; all-completed-branch Optimizer chronology; all-completed-branch Evolver chronology with Agent winners, exact Kernel artifacts, outcomes, and raw Session materialization | Retention sizing under long campaigns |
| Gateway/Agate | Implemented against SDK adapter | Scoped capabilities, sealed private Evaluation Contract, strict public Agent Problem validation, multiple immutable Agent evaluations per Attempt, exact Kernel/raw-result retention with separate opaque-id Worker projection, single-private-case Profile, authoritative full-set retention-comparator outcome, Attempt-owned jobs | Real Agate/GPU all-operation acceptance and crash-window rehearsal |
| GPU Wiki | Partial | Live frozen query, durable post-Epoch Outbox, exact bounded raw Session upload, local wire-compatible test double | Production Wiki contract/retention rehearsal |
| Token accounting | Implemented | Positive per-Session Optimizer quota; unbounded Evolver with mandatory fail-closed provider-usage report | Real provider accounting acceptance |
| Administration/operations | Implemented locally | Authenticated Campaign/Task/recovery/Event/Lineage-seed APIs; unified pre-launch-to-terminal Worker Session catalog for Optimizer, Baseline, Generalization, and Evolver with raw Trace/workspace/token/error identity; durable Kernel, evaluation, and Bootstrap Run catalogs; bounded exact source/result export; readiness; real Attempt debug shell without starting an Agent; backup-oriented SQLite; Artifact/workspace GC | TLS/front proxy, external alerting, crash drills |
| Packaging/source isolation | Implemented locally | Runtime wheel excludes Core/Evolver; exact Git import and sealed provenance | Clean-host release rehearsal |
| Whole-worker isolation | Host Sandbox and nested-bwrap outer-container mode implemented | All modes exclude private evaluator paths/data from Worker protocol. Both production modes add a read-only root/private `~/workspace`, Runtime-storage and sibling-root masking, dropped capabilities, and namespaces. Host `sandbox` adds per-Session cgroup v2 limits; `container` invokes bwrap directly and delegates aggregate resource limits to its dedicated OCI container. | Exact production image escape/resource/timeout/soak evidence for the selected boundary |

## Automated baseline

The repository maintains Runtime, separately versioned Core/Evolver, and local Wiki suites plus
Ruff, strict mypy, wheel-independence, and Linux Sandbox checks. Counts are not frozen here; release
evidence records exact command output. The Lima reference environment has exercised the Sandbox,
virtiofs Worker-root preparation, and live Claude/Codex/QoderCLI connectivity. Exact commands are in
[Testing and Production Acceptance](testing-and-acceptance.md). These checks do not establish
production safety because exact deployment-image hostile-code validation and target
Agent/Agate/Wiki/GPU crash/soak evidence remain outstanding.

## Next implementation order

1. Run and archive the whole-Worker filesystem/network/namespace/cgroup/host-service escape-negative suite on the exact deployment kernel and systemd/bwrap versions.
2. Run every authorized operation through the selected Core backend, Runtime Gateway proxy, real Agate, and target GPU.
3. Exercise forced termination at every persisted transition and rehearse backup, restore, GC, credential rotation, and Wiki Outbox recovery.
4. Run representative multi-DSL soak tests and set evidence-based storage/resource limits.
