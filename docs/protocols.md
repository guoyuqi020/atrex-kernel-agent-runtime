# Protocols

English | [中文](protocols.zh.md)

All JSON schemas reject unknown fields. IDs use typed prefixes; Artifact digests use `sha256:<64 lowercase hex>`. Durable payloads are canonical UTF-8 JSON where a digest depends on serialization.

## HTTP surface

| Route | Authority | Purpose |
| --- | --- | --- |
| `GET /healthz` | none | Process liveness. |
| `GET /readyz` | none | Local Registry, Gateway control, Agate job store, and Artifact-store probes. |
| `POST /v1/operations` | Attempt capability | Structured GPU/Agate operation; every Agent `evaluate` is an immutable exploratory record. |
| `POST /v1/runtime/queries` | Attempt capability | Unmetered reads for a known Trial, Kernel source, or Gateway result. |
| `POST /v1/runtime/journals` | Attempt capability | Immediate durable Direction/Experiment mutations and authorized list/load reads. |
| `POST /v1/wiki/query` | Attempt capability | Runtime-enriched live Wiki query, frozen before return. |
| `POST /v1/admin/campaigns/bootstrap` | Admin bearer | Campaign schema v3 only. |
| `POST /v1/admin/campaigns/{id}/lineages` | Admin bearer | Add an independently versioned Lineage from sealed Agent and Kernel content. |
| `GET /v1/admin/bootstrap-attempts/{id}/runs` | Admin bearer | Ordered physical Bootstrap execution Generations. |
| `GET /v1/admin/bootstrap-attempts/{id}/runs/{generation}` | Admin bearer | Exact Bootstrap status, failure, workspace, tokens, Trace, Report, and result identities. |
| `GET /v1/admin/campaigns/{id}/epochs` | Admin bearer | Active/Challenger competition and winner history across Campaign lineages. |
| `GET /v1/admin/lineages/{id}/epochs` | Admin bearer | Ordered Epoch winner history with `agent-vN` and `vN` labels. |
| `GET /v1/admin/campaigns/{id}/attempts` | Admin bearer | Every Attempt across Campaign lineages, including no-Candidate outcomes. |
| `GET /v1/admin/lineages/{id}/attempts` | Admin bearer | Ordered complete Attempt history for one lineage. |
| `GET /v1/admin/attempts/{id}` | Admin bearer | One Attempt's status, report disposition, and input/output version relation. |
| `GET /v1/admin/attempts/{id}/report` | Admin bearer | Runtime Final Attempt Report with authoritative parent/Candidate Gateway performance. |
| `GET /v1/admin/attempts/{id}/evaluations` | Admin bearer | Ordered Agent-exploratory and Runtime-final evaluations. |
| `GET /v1/admin/attempts/{id}/evaluations/{evaluation_id}` | Admin bearer | One immutable evaluated Kernel/result identity pair. |
| `GET .../evaluations/{evaluation_id}/source` | Admin bearer | Exact bounded files of the Kernel evaluated at that step. |
| `GET .../evaluations/{evaluation_id}/result` | Admin bearer | Exact raw Gateway result for that step. |
| `GET /v1/admin/attempts/{id}/kernel-trials` | Admin bearer | Exact unversioned Candidate snapshots observed in the Attempt. |
| `GET /v1/admin/attempts/{id}/kernel-trials/{trial_id}` | Admin bearer | One Trial's Gateway observations and Agent decisions. |
| `GET .../kernel-trials/{trial_id}/source` | Admin bearer | Exact files for a measured, probed, or reverted Trial. |
| `GET .../kernel-trials/{trial_id}/results` | Admin bearer | Retained structured results for each Trial observation. |
| `GET /v1/admin/campaigns/{id}/kernels` | Admin bearer | Every baseline and terminal Attempt Kernel across DSL lineages. |
| `GET /v1/admin/lineages/{id}/kernels` | Admin bearer | Ordered Kernel catalog for one DSL lineage. |
| `GET /v1/admin/campaigns/{id}/agent-revisions` | Admin bearer | Versioned Agent histories across Campaign lineages. |
| `GET /v1/admin/lineages/{id}/agent-revisions` | Admin bearer | Ordered `agent-vN` history for one DSL lineage. |
| `GET /v1/admin/agent-revisions/{id}` | Admin bearer | One Agent revision with parent, disposition, and Artifact provenance. |
| `GET /v1/admin/kernels/{id}` | Admin bearer | Kernel, producing Agent/Attempt context, primary evaluation, and durable repeat measurements. |
| `GET /v1/admin/kernels/{id}/source` | Admin bearer | Bounded exact Kernel Artifact files as UTF-8 or Base64. |
| `GET /v1/admin/kernels/{id}/measurements` | Admin bearer | Primary Gateway evaluation plus durable retention/promotion repeats. |
| `/v1/admin/campaigns/...` | Admin bearer | Inspect persisted Campaign/Lineage models, cancel, and schedule Campaigns/Tasks. |
| `/v1/admin/epochs/.../recover` | Admin bearer | Explicit Failed Epoch recovery. |
| `/v1/admin/events`, `/events/export`, `/events/prune`, `/metrics` | Admin bearer | Bounded observability and maintenance. |

There is no `/v1/admin/bootstrap` compatibility alias.

Each distinct Agent evaluation seals and records the exact candidate and raw result but does not
commit an Attempt outcome. Runtime seals the terminal nomination and evaluates it through the
configured authoritative retention comparator; only its `runtime_final` records are linked to the outcome. The
`list-evaluations` and `show-evaluation [--source] [--result]` CLI commands expose the same history.
Every candidate-bearing operation binds its sealed Candidate Artifact before external execution.
The binding remains a durable Artifact root after operation failure, Event pruning, or source
rollback. A schema-v12 Agent Attempt Handoff gives every Direction and Experiment durable IDs.
Direction definitions are immutable and lifecycle changes are append-only events; each Experiment
names its Direction. Ordinary comparison actions bind both `before` and `after` to an exact Kernel
Artifact, Kernel Trial, and non-empty list of Gateway Result Digests. The Bootstrap-only `baseline`
action instead requires `before=null` and one complete `after`, establishing the first measured
anchor without registering `v0`. `evidence` contains observations; `analysis` contains interpretation
and the hypothesis verdict; other actions record whether the Agent kept the After Kernel, restored
the Before Kernel, or abandoned the direction. Runtime maps those actions to the observed `after` Trial's
internal disposition. The report's terminal `status` is an Optimizer handoff state, not a retention
decision; no top-level `decision` is accepted. Runtime then joins that handoff with the registered
input/output Kernel revisions and their authoritative Gateway Result Artifacts to derive the
schema-v1 Final Attempt Report. This projection exposes exact Kernel Artifact identities and
aggregate/per-Shape latency while withholding internal Kernel Revision IDs. It also projects the
trusted content-level Production Gate as `NOT_ENABLED`, `PASS`, `FAIL`, or `NOT_RUN`; this field is
derived by Runtime and is never Agent-authored.
Direction and Experiment mutations use `/v1/runtime/journals`. Runtime validates and commits each
mutation before replying, and stores its idempotency key against the logical Attempt. Consequently
the Journal is immediately queryable and survives a physical Session failure or recovery-generation
change. `attempt-report` obtains a Runtime snapshot and embeds that same authoritative current
Attempt Journal in the terminal handoff; terminal publication is not the first persistence point.
Completed selected and discarded search paths contribute frozen Direction and Experiment journals
to subsequent Optimizer sessions, while scheduling provenance remains hidden.
These unversioned `Kernel Trial` records do not consume `vN` revision numbers.
An Optimizer can inspect a known Trial returned by a Gateway response or retained Experiment record
with `kernel-trial-show`, read exact source by `kernel_artifact_digest` with `kernel-artifact-read`, and read the normalized
Agent-visible measurement by `gateway_result_digest` with `gateway-result-read`. Runtime includes the current Attempt so
a reverted Candidate remains recoverable within the same Session, then applies the same promoted
and same-trajectory visibility rule to older Attempts. These Runtime-local reads are unmetered and
do not contact Agate.
The Worker response is deliberately different from the admin-visible raw result: private input,
request, per-case failure, and evaluator-spec fields are withheld. `evaluate` returns aggregate
correctness/latency and optional latency keyed by opaque numeric `shape_id`. `profile` accepts an
optional numeric `shape_id`, defaults to one evaluator-owned case, and returns a sanitized profiler
view labeled only with that number.
`check` exposes no Shape selector: Runtime deterministically selects the first opaque Contract
Shape and supplies its private `init_kwargs` to Agate Compile so parameterized `Model`
constructors are checked correctly.
These semantics do not depend on `launcher.mode`.

Runtime normalizes successful `evaluate` and `profile` responses into append-only internal
`gateway_measurements` rows. They remain available to Gates, frozen Evidence, and administrative
interfaces, but are not a separate Agent query surface. `kernel-trial-show` accepts a known
`kernel_trial_id` and returns only its Kernel Artifact Digest and normalized `gateway_results`;
result entries omit their already-resolved Gateway Result Digests. Visibility is derived, not
caller-selected: completed Epochs expose only winning-branch Attempts to an Optimizer, while the
current Epoch exposes only earlier completed
Attempts in the same branch, Challenger slot, and trajectory plus its own in-progress Trials.

Attempt history is independent of Kernel history. `list-attempts [--format table]` and
`show-attempt` retain `pivot`, `blocked`, infrastructure-failed, and other no-Candidate sessions;
such Attempts have no output `vN` because no Kernel revision was registered. A completed Agent
Session also seals its exact terminal Runtime State. The producing Attempt stores the Artifact as
`runtime_state_digest`; its existing `attempt_id` is the producer reference, and no separate State
checkpoint identifier exists. Serial recovery uses this immutable digest rather than trusting a
mutable workspace cache. Before the first physical Session, Runtime also seals the logical input as
`input_runtime_state_digest`. Physical retries may advance `runtime_state_digest`, but never rewrite
the logical input. Runtime uses the last terminal State of the winning branch's best-Kernel
Trajectory as the common seed for both the next Active Branch and Evolver Candidate. Missing
terminal State falls back to that Trajectory's first Attempt input.

Epoch history exposes the pre-competition Active, ordered Challenger pool, terminal winner,
`active_retained`/`challenger_promoted` decision, starting Kernel, and selected global-best Kernel.
`list-epochs [--format table]` accepts either Campaign or Lineage scope.

Kernel catalog entries expose the immutable Kernel/Agent revision IDs, Lineage-local `vN` and
parent-version labels, Campaign, Lineage, DSL, optional Epoch/Attempt/Branch context, retention
decision, source and Gateway-result Artifact references, correctness, latency, parent-relative
improvement, Gateway-reported SOL percentage, and creation time. A single Shape uses its exact
`sol.pct`; multiple Shapes use the all-Shape geometric mean. If any Shape has no SOL value, the
projection is JSON `null` and the table renders `-`. A same-allocation ABBA result carries the sealed
Evaluation Contract digest and authoritative Candidate latency/SOL aggregates, so the Kernel catalog
does not lose Roofline evidence when the ABBA Artifact replaces an exploratory Eval result. Artifact
reference objects contain `digest`, `kind`, and
`referenced_at`; Digest-only fields remain for compatibility. Exploratory evaluations use stable
`g<generation>-e<ordinal>` labels and do not consume Kernel version numbers. Framework-baseline and
Attempt evaluations come from the Kernel revision. Every repeated retention/promotion Evaluate is independently persisted in
`kernel_measurements`; it remains queryable after lifecycle Event pruning.

Kernel Agent catalogs use an independent Lineage-local `agent-vN` sequence. They expose the parent
Agent version, Bootstrap/Lineage-seed/Evolver origin, introduction Epoch, active flag, promotion disposition,
creation time, and exact Optimizer Artifact identity. Every Kernel entry includes the Agent version
that produced it. `list-agent-revisions [--format table]` and `show-agent-revision` expose the same
projection as the authenticated API.

## Bootstrap and Bundle protocols

Campaign schema v3 requires `creation_key`, operator, an Agate GPU environment selector in
`hardware_target`, Evaluation Contract
path, full `base_revision.commit`, `challenger_count`, `challenger_start_epoch`,
`trajectories_per_branch`, `attempts_per_trajectory`, and per-DSL
`baseline_kernel`/`initial_evidence`. The keys of `lineages` are the authoritative and complete
initial Bootstrap DSL selection. Optional `shape_train` is the preferred public train-domain
contract and is mutually exclusive with the legacy `agent_problem`; either bypasses Core problem
generalization. Exact `shape_valid.json` cases, `metadata.json`, and `roofline.json` are sealed in the
Evaluation Contract and never copied into the Agent workspace. The stored public-contract Artifact
remains complete. Core renders a concise execution view into the Agent Prompt: generator/range
provenance is omitted and `shape_domain` becomes the sole parameter-domain source. Fixed parameters
are direct JSON values; variable parameters use range or multi-value Domain objects.
Operation/category labels are supplied by the objective;
`operator_contract` remains only when it carries non-Shape ABI semantics such as construction,
mutation, layout, or return behavior. Directly implied invariants are removed, while cross-field and
semantic invariants remain visible. For migration-only `atrex.agent_problem.v1`, the legacy flat
`operator_contract` is normalized entirely into `shape_domain`; private `shapes.json` remains sealed
as the evaluator-case fallback. Runtime
accepts optional `lineages.<dsl>.models.optimizer` and `.evolver` model identities. A null or
missing identity selects the Backend CLI default. Optional top-level
`problem_generalization_model` is scoped only to generated Agent Problems. Runtime persists these
identities and binds them to fresh Sessions as `ATREX_AGENT_MODEL`. Runtime imports Core once and
creates one revision per selected DSL from the same optimizer digest. One
stable Bootstrap Attempt may have multiple append-only recovery Generations. Each new physical
Session receives a fresh capability generation; old Gateway operations remain keyed by their
original generation, while only a correct committed outcome can create the baseline.

Before sealing a new Campaign, Runtime calls Agate `get_env(hardware_target)`. The returned `arch`
(for example `sm_120`) becomes the Campaign/Lineage hardware target visible to every Agent. The
canonical returned `gpu` (for example `L20N`) is sealed separately as `agate_gpu` in the Evaluation
Contract and is used only for Agate scheduling. Bootstrap fails closed if Agate omits either field;
an Agent-visible hardware target is never inferred from the environment alias.
When Agate reports an explicit accelerator backend, Runtime preserves it; otherwise Runtime infers
CUDA, ROCm, or PPU from the canonical GPU and architecture. The backend and optional device slug are
sealed in the Evaluation Contract. `PPU-*` and `ZW-*` targets automatically
disable managed clock locking because PPU-SMI does not expose that operation.

Lineage seed schema v1 requires `creation_key`, fixed `dsl`, Epoch topology, and a discriminated
`seed`. `source_type: "artifacts"` names `agent_artifact_digest` and `kernel_artifact_digest`;
`source_type: "revisions"` names `agent_revision_id` and `kernel_revision_id`. Optional
`models.optimizer`, `models.evolver`, and `initial_evidence` configure the new Lineage. Runtime
revalidates the complete Agent Bundle, independently evaluates the exact Kernel under the target
Campaign contract, and publishes fresh `agent-v0`/`v0` identities. The CLI equivalent is
`seed-lineage --config ... --campaign ... --spec ...`. Standard Bootstrap remains anchored by
`base_revision.commit`; the seed operation imports no Git source and accepts only sealed CAS content.
Campaign Bootstrap also freezes the deployment-selected full Evolver Commit. Bootstrap and Campaign
inspection responses expose `evolver_commit`; all scheduling paths reject a different configured
Commit before Evolver execution. A seeded Lineage inherits this Campaign-level value.

`roofline` inside the Evaluation Contract remains optional. An explicit value is authoritative. If
it is null and the deployment configures `campaign.roofline_builder`, Runtime executes the
commit-pinned Atrex Bench converter, requires exact Shape coverage and finite non-negative W/Q/SOL
fields, and seals the generated value before problem generalization. A retry of an existing Campaign
reuses the sealed generated contract when all submitted non-Roofline fields are unchanged. Without
either source, or when construction fails, Runtime preserves the Roofline-free contract and runs
an NCU SOL Profile after every correct Agent or authoritative Eval. The two raw responses share one Gateway
Result Artifact; Profile failure does not invalidate Eval correctness or latency.

Each Lineage result retains the compatibility fields and additionally returns
`bootstrap_attempt_id`, a structured `baseline_kernel` object containing its Revision ID, `v0`
label, creation time, and Bootstrap producer identity, plus the initial `agent-v0` identity and
Optimizer Artifact. The producer is intentionally separate from
the ordinary Epoch `attempt_id` shown in the Kernel catalog.

Core `atrex-bundle.json` schema v1 declares one repository-relative entrypoint. Runtime sets `ATREX_CORE_PHASE` to `problem_generalization`, `framework_baseline`, or `optimization_attempt` and supplies phase-specific manifest/report paths, `ATREX_USAGE_UNIT`, `ATREX_USAGE_BUDGET`, a provider-usage report destination, optional Session trace, isolated home, and scoped Gateway/Wiki authority. QoderCLI uses credits; Claude, Codex, and Pi use provider tokens. Exit 0 means completed; exit 125 means the selected provider-native quota is exhausted; every other exit is explicit. A valid schema-v2 provider-usage report is mandatory.

In both production modes Runtime additionally sets `HOME=/home/agent` and
`ATREX_WORKSPACE=/home/agent/workspace`. `ATREX_SANDBOX` is `bwrap-cgroup-v2` for `sandbox` and
`bwrap-container` for `container`. The Worker shares the surrounding network and DNS directly;
Runtime does not inject proxy variables or restrict reachable ports. Both modes retain the bwrap
filesystem boundary; only `sandbox` adds a per-Session cgroup.
Any host-absolute path below the Session root is translated to its `~/workspace` equivalent before
launch.

Evolver `atrex-evolver-bundle.json` schema v1 declares one entrypoint. Runtime sends exactly `Run the
versioned Evolver Bundle once.` on stdin. Evolution input schema v10 fixes parent revision, evidence
checkpoint, DSL, optimizer digest, workspace paths, and a read-only `visible_agents` catalog. That
catalog contains the Active parent, retained Lineage Agent history, and Challengers already created
earlier in the same Epoch. Each entry includes repository path, Parent link, creator, relationship,
and current-Epoch Challenger ordinal when applicable. The unversioned Evolution output declares
proposal mode, selected Agent revision, hypothesis, expected effect, and exact Source-root-relative
changed paths.
Trace schema v9 records Bundle commit/tree/Artifact identity, selected model, and process evidence.
Evolver has no token cutoff; its mandatory report records complete provider usage with a null budget.
Schema v10 has no Runtime query authority. Evolver reads current Agents,
historical Agents, optimization summaries, latest-Epoch Conversations, and runtime state directly
from frozen files. A historical derivation copies the selected historical Source into Candidate
Source and may curate the flat common state seed from visible historical state; Runtime validates
the declared Agent revision, the reported Source Diff, and the private Runtime State Diff without a
Candidate Base side record.
The Agent submits this output through the read-only Bundle-local `evolution-report` command. Invalid
drafts return structured `issues`, `request_schema`, and `recovery` without publishing; the first
valid draft is atomically published to `scratch/evolution-report.json`. Runtime independently
revalidates the published report and Candidate after the Agent exits.
Per-Trajectory `runtime-state/trajectories/<N>/{skills,tools}/` is the only adaptive Skill/Tool
storage and belongs to non-versioned Lineage state. Top-level `skills/` and `tools/` are reserved and
invalid in versioned Agent source. Evolver may curate the flat
`candidate/runtime-state/{skills,tools}/` seed directly or change the versioned mechanism governing
future state use. Runtime seals complete Source and State for every new Revision and copies that
exact State to every new Trajectory. The Agent Revision directly records both
`optimizer_digest` and `runtime_state_digest`; the Evolution Trace is provenance, not the only
location of the State identity. Legacy revisions without the direct field remain readable through
their Trace.

## Evidence and Wiki

Attempt Evidence schema v2 is immutable and includes only earlier Attempts from the same Branch slot
and Trajectory. Completed Epoch Evidence preserves Challenger and Trajectory identities, summaries,
reports, diffs, normalized Session projections, normalized Evaluate/Profile measurements, lessons,
and source digests. Agent view schema v1 is role-scoped: an Optimizer sees the promoted completed
lineage plus bounded current same-Trajectory Attempts with branch-control identities removed.
Runtime derives the compact Evolver filesystem view from complete durable Evidence: each current
participant receives an authoritative latest-Epoch optimization summary and one Conversation per
Attempt; completed non-current Agent versions receive a summary beside their source. Older detailed
branch trees and exact Kernel Artifacts remain in Runtime's Registry and Artifact stores rather than
being duplicated into the Evolution workspace. The Evolver requires no live Gateway authority.
Authoritative Session retention omits Claude `system/thinking_tokens` estimate telemetry, and the
derived copies defensively apply the same rule to older Session Artifacts.

Each new Session Artifact contains `conversation.jsonl`, a schema-versioned, unredacted transcript
of the exact Runtime input, every retained Provider stdout event, captured Codex rollout events when
applicable, and Runtime's terminal capture status. The record states explicitly when the CLI does
not export its Provider-managed system Prompt. The high-frequency Claude
`system/thinking_tokens` estimate event is omitted from `provider/stdout.stream-json` and the
conversation, with the selection declared by `session.json.provider_event_filters`.
`events.jsonl` retains the normalized authoritative usage/projection ledger.

Wiki Query service API is version 1. Query content follows GPU Wiki's exact
`records`/`notes` projection; stable Record IDs are the `records` mapping keys and each value is the
complete safe served Record. Responses are frozen as interaction Artifacts before Core receives
knowledge-only content. Runtime does not upload Query consumption, Agent history, or Session traces
to the Wiki.

Gateway capabilities are signed, Attempt-bound, operation-limited, call-limited, expiring, revocable, and reconstructed from Registry authority. Worker-authored reports, annotations, and Agent-visible exploratory evaluations are untrusted evidence; Runtime-final Gateway outcomes and Registry transitions are authoritative.
