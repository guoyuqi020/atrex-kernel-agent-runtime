# Changelog

English | [中文](CHANGELOG.zh.md)

All notable changes to Atrex Kernel Agent Runtime are documented here.

## Unreleased

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
