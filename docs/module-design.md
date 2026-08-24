# Module Design

English | [中文](module-design.zh.md)

## Repository layout

```text
src/atrex_runtime/
  api/                     authenticated administration API and ASGI application
  cli/                     public entrypoint, parser, command families, progress output
  composition/             shared Gateway, Bootstrap, Campaign, and Wiki assembly
  asgi.py                  shared bounded-body, Bearer, and JSON response primitives
  filesystem.py            shared private-tree permission transitions
  serialization.py         canonical JSON bytes, digests, and file writes
  bootstrap.py             commit-only Campaign bootstrap
  presentation.py          shared CLI/HTTP JSON read-model projections
  lineage_seed.py          Artifact/revision-seeded independent Lineage roots
  config.py                strict deployment configuration
  dev_shell.py             Optimizer/Evolver debug shells without starting an Agent
  git_import.py            shared safe Git command/archive boundary
  roofline.py              commit-pinned trusted Atrex Bench Roofline builder
  artifacts/               immutable local CAS
  controller/              Epoch, Campaign, Evidence, fencing, durable task worker
  gateway/                 capability models/control, Agate adapter/configuration, proxy
  kernel_agents/           full-repository validation and Core Git import
  knowledge/               live Wiki Query proxy and post-Epoch Outbox
  registry/                SQLite durable state and events
  workers/                 launcher, workspaces, Core/Evolver process adapters
src/atrex-kernel-agent-core/       independent Git submodule
src/atrex-kernel-agent-evolver/    independent Git submodule
third_party/atrex-bench/            pinned trusted evaluator Git submodule
local-wiki/                         first-class local GPU Wiki service and corpus
```

## Core modules

`kernel_agents/revision.py` validates and seals a complete Core repository with limits for total files, total bytes, and entrypoint bytes. `kernel_agents/git.py` imports an exact approved commit and explicitly approved submodules without executing repository content. `git_import.py` centralizes noninteractive Git execution, bounded tar creation, path-safe extraction, and link/special-file rejection.

`bootstrap.py` supports only Campaign schema v3. The Campaign definition is separate from Runtime deployment configuration, and its non-empty `lineages` map is the sole initial DSL selection. Before any Agent Session it preserves an explicit Roofline or asks the optional `roofline.py` provider to execute the canonical Atrex Bench converter from one deployment-pinned Git commit, validates exact Shape coverage, and seals the result into the shared Evaluation Contract. It imports Core once per operation, shares the resulting optimizer digest and source provenance across selected DSL revisions, and requires either a supplied public Agent Problem or the Core problem-generalization phase. Campaign-level problem-generalization, the deployment-selected full Evolver Commit, and Lineage-level Optimizer/Evolver models are persisted as immutable resume inputs. `CampaignScheduler` validates the frozen Evolver Commit while holding a Lineage fence. A Core baseline generator is mandatory; precomputed Gateway-result and local-repository paths do not exist. `gateway/control.py` retains every Bootstrap execution in `bootstrap_runs` and keys Gateway operations by Attempt plus recovery Generation; `composition/bootstrap.py` commits terminal success or failure evidence for every caught Session exit.

Before Roofline resolution or any Core phase, Bootstrap applies Runtime `gate_policy` to the sealed
Contract. Sampling, tolerances, timeouts, full validation mode, clock policy, evaluator commit, and
Gate-owned runner controls therefore have one Campaign-frozen source of truth. The same Contract
freezes `production_gate`; `gateway/production_policy.py` applies its stateless content checks before
Worker Eval, authoritative finalization, and Artifact-seeded Lineage publication.
`composition/gateway.py` is the only constructor for that authoritative evaluator, so CLI,
Scheduler, and HTTP bootstrap cannot drift in Bootstrap stages or Production Gate wiring.

`gateway/proxy.py` treats every Worker `evaluate` as exploratory and seals both the exact candidate and raw result. `gateway/private_results.py` independently builds the Worker response: opaque per-case latency and aggregate status for Evaluate, plus recursively private-field-free Profile output. Manual Profile resolves exactly one sealed private Shape. `gateway/control.py` appends these records without occupying the Attempt outcome. `gateway/finalization.py` accepts only a nominated tree with a matching correct exploration. Bootstrap receives a fresh Runtime-owned final Eval. Optimization Attempts use the nomination only as a provisional registered Evaluation; the selected Kernel-retention Comparator replaces it with B's trusted ordinary-Evaluate arithmetic mean or ABBA geometric mean before completion. Administration routes and CLI expose the exact raw Artifacts.
`gateway/candidate.py` centralizes Kernel Artifact kind/path validation, while `asgi.py`,
`filesystem.py`, and `serialization.py` keep transport, permissions, and persisted JSON behavior
identical across the otherwise independent surfaces.

`lineage_seed.py` resolves a pair of sealed Agent/Kernel Artifacts directly or through historical
Revision IDs, validates the same-DSL complete Agent Bundle, and creates stable independent root
identities. `gateway/lineage_seed.py` performs the required destination-Campaign Agate evaluation
before `agent-v0`/`v0` publication. Source Revision links are provenance only; the new Lineage owns
an independent version tree and begins at Epoch 1.

`workers/core_phase.py` is the common command-resolution, Sandbox launch, bounded process, token-report, and Session-trace acquisition path for optimization attempts, problem generalization, and framework baseline. `workers/launcher.py` constructs the systemd cgroup plus bwrap mount/process boundary, maps every host Workspace path to `~/workspace`, deliberately retains host networking, serializes concurrent host checks, and asks systemd to create workspace roots/probes directly as the non-root Worker for virtiofs-safe ownership. `workers/core.py`, `problem_generalization.py`, and `lineage_bootstrap.py` own only their phase-specific environment and output schemas.

`registry.worker_sessions` is the unified physical-process catalog for Optimizer Attempts, Framework Baselines, Problem Generalization, and Evolver runs. A running row is committed after workspace preparation and before authority acquisition or process launch. Exactly one terminal update records completion, failure, or timeout together with the unmodified sealed Session Trace when available, provider token usage, process status, diagnostics, and stable workspace/run identity. Context columns are nullable because Generalization precedes Campaign creation; existing Attempt/Epoch identities are automatically expanded to Campaign/Lineage context. The catalog complements role-specific protocol records such as `attempt_session_traces` and `bootstrap_runs`; it does not replace their domain evidence.

`dev_shell.py` reuses the exact launch preparation in `workers/core.py`. It materializes a real
workspace for an existing or newly created first Active Attempt, issues the same scoped capability,
and starts interactive `zsh/bash` without executing the Core entrypoint. The debug entrypoint holds
the lineage fence and retains the workspace plus running Attempt on exit; it creates no fabricated
Agent trace, token report, or outcome.

The same module also reconstructs an Evolver workspace for one required Lineage ID and existing
absolute Epoch number. That snapshot uses the Epoch-recorded parent Agent and Evidence checkpoint,
includes earlier Kernel history and any Challenger already attached to the selected Epoch, and
excludes Kernels produced by the selected or later Epochs. It prepares the same Evolver environment
but creates no Challenger, Agent execution, promotion, or token report.

`workers/evolution.py` creates a fixed input/parent/agents/evidence/runtime-tools/candidate/scratch
workspace and accepts same-DSL full-repository changes. It freezes exact Lineage-local Agent/Kernel
version catalogs and every historical Kernel Artifact. `workers/evolver_tools.py` is copied into the
workspace as a bounded frozen-snapshot inspection client plus the single constrained
`candidate-reset` mutation. The latter admits only completed Lineage history, atomically replaces
the Candidate, and writes the base record checked during sealing. Runtime launches the
commit-pinned Evolver with one fixed stdin instruction. Provenance records commit, tree, sealed
Artifact digest, argv digest, environment-key names, process result, token usage, session trace,
output annotation, and candidate digest. Provider, model, and prompt selection belong to the Evolver
repository, not Runtime configuration.

`controller/epoch.py` implements the configurable Epoch topology. It creates zero or more
Challengers sequentially, exposes the growing Lineage Agent catalog to each Evolver call, executes
independent Trajectories concurrently within each Branch, and executes Attempts serially within a
Trajectory. `controller/attempt_evidence.py` isolates incremental memory by Trajectory;
`controller/evidence.py` publishes the complete measured Epoch only after selection.

`controller/projection.py` emits bounded normalized summaries that contain only a Session source digest; it deliberately excludes the unredacted `conversation.jsonl` from semantic projection. `workers/session_trace.py` applies the authoritative retention policy before every Core or Evolver Session Artifact is sealed: high-frequency Claude `system/thinking_tokens` estimate telemetry is removed from Provider stdout and the conversation, while authoritative usage remains in `events.jsonl`. `workers/evidence_view.py` follows the digest to materialize a derived read-only Agent copy and defensively applies the same policy to older Artifacts. `knowledge/ingest.py` independently constructs bounded retained Session upload projections after epoch completion and applies the same compatibility filter.

## Stable interfaces

The replaceable seams are Python Protocols: Artifact store, Registry, measurement runner, comparator, Optimizer/Evolver runner, Worker launcher, lineage lease manager, and Wiki query client. `BwrapSandboxLauncher` is the production implementation. `CleanEnvironmentLauncher` remains only behind explicit `launcher.mode=development` and makes no isolation claim. Normal Optimizer/Evolver sessions, Bootstrap/Generalization phases, and both dev-shell entrypoints use the same configured launcher.

All durable identities are content- or creation-key-derived and every externally supplied digest/ID is validated. Registry transitions are idempotent, lifecycle events are append-only, and scheduler writes use renewable generation fencing. Kernel retention and Agent promotion independently select ordinary Evaluate or same-allocation ABBA. The latter uploads a commit-pinned evaluator and both Kernel snapshots in one trusted dev Job per Shape batch, validates the exact interleaved schedule, and stores every run as a Kernel measurement.
