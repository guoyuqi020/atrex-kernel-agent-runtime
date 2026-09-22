# Changelog

English | [中文](CHANGELOG.zh.md)

All notable changes to Atrex Kernel Agent Runtime are documented here.

## Unreleased

- FA4 and FA4-SM120 now pin Evolver `794219bc`, whose Evidence contract accepts and validates the
  Runtime-projected `input/evidence/review/` directory. This prevents every Evolver-backed arm from
  failing at startup when it reaches Challenger construction. Both task definitions use new
  creation keys so regenerated workspaces cannot resolve to the incompatible frozen Campaigns.
- Evolver failure Artifact v6 retains the bounded process return code, stdout/stderr, optional
  Session trace, and optional provider usage even when the provider-usage report is missing or
  invalid. A nonzero process exit is now reported as the primary failure instead of being obscured
  by the secondary usage-report error; the bounded exception message is retained for diagnosis.
- Bootstrap Sandboxes no longer mount the repository's pinned upstream reference projects. The
  Bootstrap workspace and prompt now expose only the task seed, Agent Revision state, Runtime tools,
  and scratch space. Newly generated Runtime configurations omit `reference_projects_root`; the old
  field is accepted only to resume an existing frozen Campaign and has no effect. Runtime retains an
  empty `reference/` protocol placeholder so commit-pinned Agent Revisions created before this
  change can still start, but no Reference Project contents are mounted there.
- Worker startup failures that occur before Session capture or provider-usage reporting now preserve
  the bounded process exit status and stderr/stdout diagnostic in the Runtime error instead of being
  obscured by the secondary missing-usage-report validation failure.
- Bootstrap now materializes the configured initial-evidence Artifact at read-only
  `input/evidence/` and injects its bounded UTF-8 `README.md` into the framework-baseline prompt.
  Task hints are therefore visible to the model rather than serving only as a Registry identity.
- Runtime-owned final evaluation now uses the complete sealed Valid+Test Shape contract. Agent
  Evaluate/Profile remains Valid-only, and Agent-facing result projections continue to expose only
  opaque Valid Shape IDs and Valid-only latency metrics. Bootstrap v0 therefore cannot be
  registered without passing the private Test Shapes.

- Single-file Agent and authoritative ABBA now use Agate's native Eval ABBA API. Runtime maps two
  side measurements to one complete A/B/B/A block, rebuilds its existing authoritative aggregate
  from the returned raw SDK runs, and keeps Shape batching, retry, cache, Registry and promotion
  semantics unchanged. Multi-file source trees continue to use the Dev ABBA driver.

- Agent Revisions may now carry executable `workflow/main.py` Epoch orchestration. Runtime runs the
  Active Revision's program through the Worker isolation boundary. Agent code implements exactly one
  `run_epoch(epoch)` function over Pools and synchronized rounds; the SDK hides Attempt ordinals and
  the wire protocol while still permitting result-dependent Kernel/State routing. Runtime enforces
  the exact budget, makes round replay idempotent, and retains cross-Epoch scheduling, evaluation,
  Gate, promotion, rollback, and Registry authority.

- Production and ablation Lineages now use Runtime-owned arm templates to construct their initial
  Agent Revision. Only the selected program is sealed as `workflow/main.py`; unrelated arm programs
  are not visible to Optimizer or Evolver. `evolve-3`, Isolated, Retained, Pool-3, and
  Pool-Retained-3 still execute distinct versioned programs while sharing the controlled Optimizer
  source and exact Bootstrap Kernel.

- Default to the official remote Agate service (`https://atrex-gateway.alibaba-inc.com`, GPU
  `L20N`). Preparation and CLI examples retain explicit endpoint/GPU overrides and require the
  remote service's AK/SK credentials.

- Full Agent Evaluate and authoritative ABBA now execute one measurement per Shape instead of
  three complete calls with per-Shape medians. Inner GPU benchmark sampling and configured ABBA
  schedules are unchanged; result metadata reports `single_measurement` with one repetition.

- New Campaigns seal a fixed-seed (`42`) random 50/50 Valid/Test Shape split. Agent operations and ordinary
  evaluation use Valid only; authoritative Runtime ABBA uses Valid + Test. Agent Evidence exposes
  no Test rows, full-set latency aggregates, or Test error metrics. Odd extras go to Valid;
  single-Shape tasks are rejected. Each subset randomly samples at most 15 Shapes; excess Shapes
  are excluded from evaluation, with matching metadata/Roofline subsetting. A private `shape_split`
  record seals the seed, algorithm, source population, selections, and stable opaque Agent Shape-ID
  map. Agent requests and historical projections use contiguous `0..V-1` IDs, so source-ID gaps do
  not reveal Test membership. VecAdd examples now include two Shapes.

- Evolver now resumes one native conversation per Lineage/Backend across Evolutions, sequential
  Challenger construction, infrastructure retries, and controller restarts. Each invocation still
  receives fresh inputs and a Candidate; traces and usage exclude already recorded history.

- Clarified that Evolvers can discover new Agent capabilities from completed Optimizer
  Trajectories, independently of reviewing previous changes, and add or modify Candidate code
  only when observed behavior supports a concrete optimization benefit.

- Evolvers now review the previous evaluated Agent changes before editing again, tracing new
  Tool discovery, execution, and use against expected effects without assuming Branch victory
  proves effectiveness or an unevaluated proposal has failed.

- Added a Runtime-injected next-Optimizer service catalog for Evolvers, with guidance to compose
  existing services in Candidate code before reporting capability gaps, without granting new
  Runtime permissions or executing service calls during Evolution.

- Retired Bootstrap/Evolver Direction suggestions. Live `suggest` actions and Evolution
  `suggested_directions` fields are rejected with repair guidance; historical Journals remain
  readable. Evolver now focuses on cross-Branch evidence reconciliation and Agent improvements,
  while Optimizers choose their own Directions.

- Aligned the FA4 source-tree task with the production ablation: Epoch 1 now runs same-Agent
  replicas, preparation freezes eleven control arms (including paired Challenger-only
  Isolated-Evolve/Retained-Evolve controls and an Active/Challenger Isolated-Pool-Evolve control),
  and the task runner can launch all twelve Campaigns with 15 Optimizer Attempts per Trajectory.

- Automatically upload oversized Dev file maps through Agate OSS, including both source-tree
  ABBA paths. A checksum-verified archive restores exact files before execution; upload stages
  retry independently without changing logical request identity or measurement policy.

- Added Agent `evaluate.comparison` with `method="abba"`, using two workspace Kernel files
  (or directories). Core and KDA upload A/B sources; Runtime seals both,
  uses its pinned evaluator and shared input contract, and records per-side measurements and
  relative speedup without
  changing Kernel-retention or Agent-promotion authority. There is no standalone `abba` operation.
- Added custom input generators and Shapes to Agent `evaluate`, plus `mode="correctness_only"`
  without performance measurement or automatic profiling. Core supports workspace file helpers;
  Runtime seals exploratory requests/results without allowing them to replace trusted-contract
  evaluation for Candidate submission. Incomplete checks remain retryable.
- Added `ablation-evolve-1` and `ablation-evolve-5` with 15 x 1 and 3 x 5 schedules; the existing
  main arm is labeled `evolve-3` (5 x 3). All total 30 Optimizer Attempts, with 14/4/2 Evolutions.
  Epoch 1 runs the same Agent on two isolated Branches without calling Evolver or creating a new
  Agent revision; normal evolution starts at Epoch 2. Arms reuse the same Bootstrap baseline,
  inherit models and Evolver commit, and retain independent histories and Skills/Tools.
- Added `ablation-pool-1` and `ablation-pool-5`, and renamed the default Pool to `ablation-pool-3`.
  Every Trajectory runs 15 post-Bootstrap Optimizer Attempts. All three Pools run two parallel Trajectories
  (30 total per arm). Retained now pairs with Isolated as independent `ablation-retained-01/02`
  Campaigns by default: one Trajectory and 15 Attempts each, retaining only their own Skills/Tools.
  Arms derive their target Epoch from
  their serial Attempt grouping; Bootstrap is never counted.
- Added a required `contributing_kernel_trial_ids` field to the Optimizer's Attempt Report, naming the
  historical Kernel Trials whose code or approach the Attempt drew from. Kernel Trial identifiers are
  used because the Optimizer has no Kernel revision vocabulary and must not gain one. Both Core and
  Runtime check its shape; neither resolves it against visible history, matching how every other Kernel
  Trial reference in the report is treated. Runtime carries it into the derived Final Report, so later
  Attempts and the Evolver read it without further work.
- Told the Evolver it may study, summarize, and combine Source, Skills, and Tools from several visible
  Agents into one Candidate, and added a required `contributing_revision_ids` proposal field declaring
  every revision other than the Source base it drew content from. Runtime revalidates each credited
  revision against frozen visibility, the Lineage DSL, and completed history, then records them in the
  sealed Evolution trace, the sealed-proposal event, and Epoch lessons, and projects them as Source
  paths into `input/evolution-reports/evo-N.json`. The Source base, the Source diff target, and
  revision parentage all remain single.
- Added the exact Source change set of each prior Evolution to
  `input/evolution-reports/evo-N.json`, so an Evolver no longer has to diff two Source trees to learn
  which files that Evolution touched.
- Exposed every completed Epoch branch to the Optimizer, including the ones that were not selected,
  under `epochs/N/branches/<label>/`, with each Epoch's `summary.json` naming the selected branch. The
  current Epoch still shows only the Attempt's own Trajectory and never a concurrently running sibling.
- Consolidated the release documentation around Architecture, Configuration, Interfaces,
  Evaluation, Operations, and durable Protocols; removed superseded design/status documents and
  synchronized terminology with the current Campaign, Lineage, Epoch, Branch, Trajectory, Attempt,
  Kernel Trial, Kernel Revision, and Agent Revision models.
- Added a concise Design Principles guide explaining the separation between evolvable Agents and
  trusted Runtime authority.
- Removed a dead full-snapshot Agent State validator and consolidated shared Gateway result
  projection, Candidate-path resolution, Artifact file indexing, and SQLite transaction handling.
- Added persistent production control-plane and managed multi-DSL Campaign task scripts with
  per-DSL inspection.
- Made Sandbox host preparation concurrency-safe and Lima-virtiofs compatible by creating Worker
  roots/probes directly as the configured non-root Worker.
- Documented shared-host networking as the explicit Worker network boundary.
- Removed high-frequency Claude `system/thinking_tokens` estimate telemetry from authoritative
  Session sealing and Agent Evidence while retaining final usage records.
- Removed GPU Wiki feedback generation, persistence, delivery, and ingestion; GPU Wiki is now a
  query-only external knowledge service.
- Made new production Campaign preparation reject dirty Core or Evolver worktrees so commit pins
  always identify the exact Agent Bundle source.
- Added pinned upstream GPU kernel projects as a `reference/` tree in the framework-baseline
  workspace, bound read-only from `reference_projects_root` in both bubblewrap launcher modes. An
  Attempt no longer receives the tree: reading whole upstream projects belongs to establishing a
  first implementation, while an Attempt should act on its own measured history.
- Raised the Attempt manifest to schema 9 and stopped publishing the workspace layout in it. The
  layout is fixed at both ends and stated in the Agent Prompt, so serializing it only compared one
  hardcoded table against another while making every layout change a breaking protocol bump. A
  Kernel Agent revision registered against an earlier schema no longer starts, so existing Lineages
  must be re-bootstrapped.
- Fixed Artifact sealing silently dropping an empty directory. A runtime-state seal validated
  `skills/` and `tools/` locally, but the manifest recorded only files, so an Agent that saved no
  Skill produced an Artifact without `skills/` and the Evolver then rejected the winning
  trajectory's state at the next Epoch. The manifest now records childless directories, and omits
  the key entirely when there are none so every previously sealed Artifact keeps its digest. A seed
  missing one of the two directories is accepted rather than rejected, because the payload is
  immutable and every consumer already recreates both.

- Removed the unreachable Gateway `submit` and `sol` operations. Neither was bound in the Agent
  request registry nor offered by the deployment operation allowlist, so both were dead protocol
  surface. SOL profiling is unchanged and still reached through `profile` with `level="sol"`.

## 0.1.0 - 2026-08-20

- First release candidate of the single-node trusted Runtime.
- Commit-pinned Core/Evolver import, Campaign Bootstrap, Artifact-seeded Lineages, configurable
  Epoch topology, Agent/Kernel version histories, and resumable scheduling.
- Exploratory Gateway operations, authoritative ordinary-Evaluate or same-allocation ABBA gates,
  Production Gate, hidden Evaluation Contracts, Roofline construction, and NCU SOL fallback.
- Live GPU Wiki query with freeze-before-return.
- Claude, Codex, QoderCLI, and Pi Backend bindings with raw Session and provider-token accounting.
- Development launcher plus Linux bubblewrap/cgroup-v2 sandbox with shared host networking.
- Authenticated administration API, CLI inspection, recovery, Events, Tasks, and offline retention.
