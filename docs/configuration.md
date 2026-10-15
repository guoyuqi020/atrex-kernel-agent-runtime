# Configuration Reference

English | [中文](configuration.zh.md)

Runtime accepts strict JSON and rejects unknown fields. Local relative paths are resolved from the
file that contains them. Use checked-in examples as syntactically complete templates; this document
explains ownership and invariants rather than duplicating every numeric value.

## Configuration documents

| Document | Schema | Purpose |
| --- | --- | --- |
| [`runtime.example.json`](../runtime.example.json) | Runtime v1 | Deployment, services, policy, Agent backends, storage, and launcher. |
| `examples/*/campaign.json` | Campaign configuration | Operator, private/public contracts, Agent commit, DSL Lineages, models, Workflow command, and Runtime resource envelope. |
| [`lineage-seed.example.json`](../lineage-seed.example.json) | Lineage Seed v1 | Add one independent Lineage from sealed Agent/Kernel content. |
| Ablation Arm spec | Ablation v1 | Create a control Campaign from a source Lineage's Bootstrap baseline with an independent evolution schedule. |

There is no separate Bootstrap configuration. Manifests, reports, traces, Evidence, journals, and
usage files are Runtime outputs, not operator configuration.

## Runtime v1

### `server`

ASGI `host` and `port`. In `container` and `sandbox` modes,
`campaign.gateway_proxy_url` must identify this exact socket.

### `storage`

Four distinct locations: Registry SQLite, Gateway-control SQLite, Agate-job SQLite, and immutable
Artifact root. Put them under dedicated private directories and back them up consistently.

### `administration`

Optional admin bearer-token environment name, request/event limits, and durable Task
lease/heartbeat/poll policy. The lease must exceed twice the heartbeat period.

### `gateway_proxy`

Attempt-capability signing-key environment name, request/Candidate bounds, per-DSL changed-path
allowlists, and whether Candidate source must change. The signing key must be stable across planned
Runtime restarts.

### `agate`

Agate base URL, authentication mode (`none`, `token`, or `ak_sk`), credential environment
names, request/wait timeouts, and health interval. Runtime config names secrets but never contains
their values.

Runtime waits for Agate Jobs with bounded `get_job(wait=False)` requests and asynchronous
polling gaps (1/2/4/8/10 seconds while status is unchanged; reset on a status change).
`wait_timeout_s` bounds an individual status request, capped at 30 seconds; it does not
end the logical tool call. Evaluate, ABBA, Profile, Check, Disassemble, Dev, Bootstrap and
authoritative measurements still wait for a terminal result before publishing evidence.
Transport retries retain the 5/10/20/40/60-second policy, respect `Retry-After`, and sleep
asynchronously. Retries keep the submission key, Job ID and successful OSS upload stages.
When a later evaluation invocation must reserve and upload sources again, its wire idempotency
key incorporates the new OSS references. This avoids conflicting with an earlier upload's
request while preserving the logical evaluation identity and completed-result cache.

SDK requests use separate per-event-loop capacity limits: ordinary requests 32, OSS operations 8,
environment metadata 4, cancellation/health 4, and local archive preparation 4. These are independent
of AnyIO's default 40 worker slots; no slot is held between polls or during Runtime backoff.
The installed synchronous SDK is unchanged. A Runtime-local subclass disables its nested
submission-overload sleep, while public `transport_retries=0` disables its transport sleep.
Only bounded individual SDK calls still run in threads; cancellation waits for an in-flight call
to return before cleaning its files, and stops subsequent polls/retries. Legacy synchronous
provisioning entry points remain available; the service/evaluation paths use the async boundary.

### `kernel_agent`

Agent Bundle and public-contract size limits. Optional `base_source` identifies an initialized local
Optimizer checkout, Git executable, command/archive bounds, and explicitly allowed submodule source
identities. Bootstrap supplies a full commit SHA and archives that exact local commit without a
fetch or other network access; branches and tags are not persisted identities.

### `gpu_wiki`

Optional query-only service URL, bearer-token environment name, timeout, and request/query/response
bounds. Runtime freezes queries before returning results. There is no feedback, upload, or Outbox
configuration.

`enabled` defaults to `false`. Set `gpu_wiki.enabled: true` to expose `wiki-query` to Bootstrap and
Optimizer sessions (including dev shells), issue Attempt-scoped Wiki authority and enable the
Runtime `/v1/wiki/query` endpoint. The live tool contract and conditional Agent instructions use
the same switch. This is independent of `optimizer.tool_modules` (Direction/Experiment journals).
An absent/null `gpu_wiki`, omitted `enabled`, or `enabled: false` hides the tool and instructions,
issues no Wiki authority, and leaves the endpoint unavailable without loading Wiki credentials.
Keep the service URL and limits when toggling the flag; no second Gateway-operation setting is
needed. Evolver can see the next Optimizer's tool availability, but receives no Wiki capability.

For example, keep the other `gpu_wiki` fields and change `"enabled": false` to `"enabled": true`
in [runtime.example.json](../runtime.example.json). Restart Runtime and the campaign worker; newly
created workspaces receive the new immutable tool contract. Existing or resumed workspaces retain
their frozen contract until a new workspace is created. Existing sealed Agent revisions must contain
the opt-in Wiki tool implementation: changing a deployment flag does not update pinned Agent
source or a session that is already running. With the switch off, the API also rejects previously
issued Wiki capabilities after the service restarts. The Wiki service itself is a separate process;
select its internal corpus with `local-wiki/configs/internal.example.json`.

Agent input is only `{"query":"a focused hardware/optimization question"}`. The tool supplies
Attempt identity, idempotency and authorization; Runtime supplies trusted operator/DSL/hardware
context and freezes the response before returning it. Treat historical Wiki statements as scoped
references to test, not evidence that this Attempt's kernel is correct or faster.

### `campaign`

Deployment policy shared by every scheduled Campaign:

- four separate Worker roots for Attempts, Evolution, problem generalization, and Bootstrap;
- Lineage fencing lease/heartbeat;
- Runtime Gateway URL, allowed operations, call quota, and capability lifetime;
- `gate_policy`, Kernel-retention comparator, and Agent-promotion comparator;
- infrastructure retry count, Bootstrap Lineage concurrency, and Branch concurrency;
- optional trusted Roofline builder;
- Evidence projection bounds;
- Optimizer and Evolver worker policy;
- launcher mode and isolation policy.

DSL topology, model identity, and `K/Y/X` Epoch shape belong to the Campaign definition, not Runtime service configuration.

`campaign.optimizer.output_limit_recovery_retries` controls fresh Core/KDA conversations after a
classified output-limit failure. It defaults to `2`, accepts integers `0..10`, and is independent
of the report-only `report_completion_retries` allowance. Recovery preserves the Attempt workspace
and remaining time/usage budgets. See [output-limit recovery](interfaces.md#recovery-after-an-output-limit-failure).


## Gate and comparison policy

`campaign.gate_policy` is the trusted source for:

- Optimizer correctness cases, benchmark iterations, and exploratory Evaluate repeats;
- ordered Bootstrap stages and Bootstrap benchmark iterations;
- retention correctness cases and benchmark iterations;
- Production Gate enablement;
- warmup, `atol`, `rtol`, timeouts, and clock locking;
- commit-pinned Atrex Bench evaluator import limits.

Runtime overwrites Gate-owned fields in the input Evaluation Contract before sealing it.

`kernel_retention_comparison` and `agent_promotion_comparison` each select:

- `method: "evaluate"`: `repeats` plus `measurement_uncertainty_us`;
- `method: "same_allocation_abba"`: `repeats`, minimum improvement, allocation timeout, Shape
  batch size, and maximum parallel Shape batches.

The ABBA timeout must fit its full interleaved schedule. See
[Evaluation and Promotion](evaluation.md).

### Roofline builder

Optional `roofline_builder` pins one Atrex Bench repository and full commit, Git/Python
executables, fetch/execution/output bounds, and optional Agate-target-to-SKU mapping. It runs only
when the Campaign Evaluation Contract has no explicit Roofline. A generated Roofline must cover
exactly the private Shape IDs before Runtime seals it.

## Worker policy

### Optimizer

`campaign.optimizer` selects `claude`, `codex`, `qodercli`, or `pi`, reasoning effort,
opaque Backend settings, command prefix, explicit environment inheritance, isolated-home keys,
trace/usage paths, Attempt and Bootstrap timeouts, termination grace, diagnostic/report bounds,
and per-Session usage quota.

QoderCLI uses provider credits; other supported Backends use provider token buckets. Core must emit
the provider-native terminal usage report. Bootstrap uses `bootstrap_timeout_seconds`; ordinary
Attempts use `timeout_seconds`. Both are active-time budgets: Runtime excludes time spent
servicing authenticated Gateway, Journal, history and Wiki requests for that Session. Concurrent
requests contribute the union of their waiting intervals, not their summed durations; another
Session's calls cannot pause this Session. Token/credit quotas are unchanged.

Runtime owns `.runtime/session-clock.json`, mounted read-only inside production Agent sandboxes.
The outer process watchdog and current Core/KDA provider/continuation watchdogs read the same clock.
Dead Runtime request processes release OS leases so an abandoned request cannot permanently
pause the budget. Existing frozen Core/KDA bundles must be refreshed to gain the inner-clock support.

Runtime-bound Core/KDA Bootstrap and Optimizer Sessions hand Runtime CLI HTTP requests to an
independent supervisor. It stops the Provider's owned process groups (including subagents) before
sending the request, and resumes them only after all pending calls and Runtime's trusted in-flight
state have cleared. Shell `timeout`, background execution, and a killed CLI cannot release this
barrier. Runtime and its HTTP worker continue running. Explicit Session cancellation and policy
termination still stop the Agent and clean up its HTTP workers.

Claude also uses a `PreToolUse` hook to keep Runtime CLI Bash calls in the foreground with at least
the 72-hour Gateway HTTP allowance. Other Bash calls retain their normal behavior outside Runtime
waits. HTTP read timeouts reconnect with the same request identity. A killed CLI may lose its output;
after the request ends, an identical retry uses the existing recovery/deduplication contract.
Refresh frozen Core/KDA bundles to enable the barrier; updating only the service is insufficient.

`campaign.optimizer.tool_modules` independently enables the `directions` and `experiments`
Runtime tool families. It defaults to `["directions", "experiments"]`; use either one-element
array or `[]` to omit the other family or both. The setting applies to Bootstrap and optimization
Attempts. The live Session contract exposes only enabled commands. With Directions alone, closures
need no Experiment support IDs; with Experiments alone, `record-experiment` omits `direction_id`.
Session Prompts follow this setting: they retain the common workspace and evaluation guidance,
but include instructions only for enabled Journal tools. The default setting keeps the original
Prompts unchanged. Existing Sessions keep the contract and Prompt they started with.
Without Experiments, an unchanged visible historical Kernel can still be nominated with
`candidate_ready`: Runtime automatically validates and reuses its matching successful ordinary
full Evaluate evidence. The operator, hardware, DSL, sealed Contract, and exact Kernel content must
match; custom inputs, correctness-only checks, Profile, and exploratory ABBA do not qualify.
No Experiment or duplicate Evaluate is required, and the original measurement identity is preserved.
A failed current full Evaluate cannot be overridden by historical success. With Experiments enabled,
historical reuse continues to require an explicit `record-experiment` with `action="adopt"`.

A `LineageSeedSpecV1` or `AblationArmSpecV1` may set `result_artifact_read_enabled: false`
(default `true`). This frozen per-Lineage setting removes `result-artifact-read` from the live
Optimizer contract and Prompt and rejects direct HTTP reads. Use `tool_modules: []` alongside it
for the `no-result-read` ablation. Legacy Kernel Trial query tools are already unavailable.
Measurements still return in the originating tool response and remain stored for human analysis.
Reports, conversations, source reads, the Pareto frontier, other enabled queries, and the Agent’s
own notes remain available. Transport replay with the same idempotency key is unchanged; a new
request for an identical completed task returns its visible cached response without a GPU call or
new measurement record. Invisible peers remain isolated and get separate measurements. Arms with
result reads enabled retain the ordinary completed-duplicate error. Running duplicates wait.
Bootstrap and existing Lineages retain their defaults.

A frozen `dev_only: true` Lineage/ablation setting additionally requires `tool_modules: []`
and `result_artifact_read_enabled: false`. Agent Gateway access is restricted to `dev`. Kernel source reads
remain available within the existing history visibility rules; Result Artifact, Pareto, Journal
and Wiki queries are disabled, including direct HTTP calls. The live contract exposes
`gateway-execute` (Dev), `kernel-artifact-read`, `attempt-report`, and local `runtime-contract`
discovery. Internal report-receipt recovery remains available. Agent-owned scripts may implement
custom checks, benchmarks, profiling and environment inspection through Dev. Their output is
exploratory evidence, never an authoritative Eval verdict.

This arm accepts `candidate_ready` without a preceding Agent Evaluate. Runtime always performs
its independent private validation before the normal retention gate, including when other arms
hand off Agent measurements directly to ABBA. Existing report schema, source restrictions,
filesystem history visibility, budgets, and Bootstrap behavior are unchanged. This is not a
no-memory arm: selected historical reports/conversations and the Agent's notes remain available.
`dev-only` is the sixth setting: shared plans now have fourteen arms (seven per sharing mode), and the
FA4/L20D, GDN/L20D and Qwen3.5 FA/L20N independent plans have twenty-one Lineages (seven × three repeats).


`raw: true` is the seventh ablation setting. It requires `dev_only: false`,
`tool_modules: []`, and `result_artifact_read_enabled: false`. It exposes **no Runtime
commands**, including source reads, local contract discovery and report handoff; the server
also rejects all operations from Raw Attempts. Only Raw Sessions receive the deployment's
configured Agate endpoint/auth as `AGATE_URL` and `AGATE_TOKEN` or `AGATE_AK`/`AGATE_SK`.
Use `"$ATREX_AGATE_PYTHON" -m atrex_gateway_client.cli --help` or the SDK directly. The
launch environment must have the Agate client installed. Uploads, polling, retries,
correctness checks and timing methodology are Agent-owned; Runtime call quotas, duplicate
handling and tool-wait budget exclusion do not apply to direct calls. Session token and
elapsed-time budgets still apply. Raw Sessions do not use Runtime report/output-limit
recovery prompts or automatic historical summaries.

No Bootstrap reports, previous conversations, Epoch summaries, Journal/Kernel Trail,
peer notes or source/result query APIs are supplied. Only self-authored files under `tools/`
(including notes, source copies and saved responses in any format) carry along the same
trajectory, regardless of which trajectory wins. Scratch is temporary. The next working
kernel still follows the selected Workflow's kernel-routing policy; sharing modes do not
turn on experience sharing for Raw. On normal exit, the controller seals `work/kernel`
and privately evaluates it with the same acceptance policy. It creates only an internal
handoff receipt, with no invented Agent findings; audit traces/tokens and scores remain
available to human analysis but are never mounted as Raw Agent history. Private evaluation
creates a revoked, tokenless Gateway subject for database references; it grants no Runtime
tool authority to the Raw Agent. Bootstrap itself
keeps its existing rules. Deployment credentials are injected into the process environment,
not written into the contract, prompts or reusable files.

### Evolver

`campaign.evolver` additionally pins the Evolver repository and full commit plus import/Bundle
bounds. It records provider-native usage but has no token/credit quota. Its process remains bounded
by `timeout_seconds`.

Campaign Bootstrap freezes the Evolver commit. Resume, scheduling, and debug-shell operations reject
a Runtime config that selects a different commit.

### Environment

`environment.values` contains static non-secret values. `inherit` requires named variables from
the Runtime process; `inherit_optional` copies them only when present. Unknown ambient variables
are not forwarded. Do not configure isolated-home keys such as `HOME` through these maps.

## Launcher

`campaign.launcher.mode` is one of:

- `development`: clean environment and lightweight debug boundary; trusted local use only.
- `container`: direct bubblewrap inside a dedicated outer OCI container.
- `sandbox`: bubblewrap launched through systemd with per-Session cgroup v2 limits.

`backend_credentials` controls read-only projection of selected CLI login state. Both production
modes start with an empty filesystem and share an explicit dependency allowlist: OS binaries and
libraries, public certificates/resolver data, the active Python environment, the Runtime Python
package, selected Provider binaries/credentials, and the current Agent workspace. Extra read-only
binds cannot include Runtime storage, sibling workspace roots, or their ancestors. Repository
`data`, prior experiments, archives and other home files are absent by default.

Systemd mode mounts a fresh PID-namespace procfs. Container mode projects only this Worker's
namespace-local process metadata through a trusted parent, retaining Core's detached-subprocess
suspension and cleanup without exposing host process `root`, `fd`, `cwd` or `environ`. A fixed
`exe` alias points to the allowlisted Provider binary for standalone Bun. No host procfs directory
is mounted. A small libc compatibility shim obtains only the calling process's read-only memory
map from a peer-credential/ancestry-checked local broker; Agents cannot request a path or another
PID. It lets standalone Bun initialize with accurate stack bounds without host procfs. Container
hosts require a C compiler (`cc`) to build this cached shim; no GPU toolkit or Torch is required.
The legacy `reference_projects_root` setting is ignored.

`sandbox.resources` sets memory, swap, CPU quota, and PID limits. Container mode has no per-Session
resource block because the outer container owns aggregate limits. Both modes preserve the
surrounding network namespace and therefore allow public egress and reachable host services.

## Evidence

`campaign.evidence.max_trace_files` bounds only Runtime-normalized JSONL ledgers selected for
semantic projection; Provider-native main and child-Agent transcripts remain opaque and do not
count toward it. `max_trace_bytes` bounds the complete Session Artifact, including those Provider
files. The remaining limits bound projected events/text and Kernel diffs. Redaction patterns apply
only to normalized projections. Original Session Artifacts remain immutable and may be materialized
into authorized Agent Evidence or retrieved by administration; they are never uploaded to GPU Wiki.

## Campaign configuration

Required top-level fields:

- `creation_key`, `operator`, and Agate environment selector `hardware_target`;
- private `evaluation_contract`;
- exactly one public input: preferred `shape_train`, legacy `agent_problem`, or neither when Core
  problem generalization is configured;
- full `base_revision.commit`;
- resource bounds `max_challengers` and `optimizer_attempt_budget`;
- non-empty `lineages` map keyed by DSL.

Each Lineage provides optional Optimizer/Evolver model identities and `initial_evidence`, plus
either `baseline_kernel` or `source_manifest` + `source_repository` for a commit-pinned multi-file
Kernel. See [Source trees](source-trees.md) for the manifest and immutable edit scope.
Missing model identity delegates to the configured Backend CLI default.
Optional `problem_generalization_model` applies only when Runtime invokes Core problem
generalization.

These two values are limits, not an Epoch topology. The versioned `workflow/main.py` decides when
to replicate or evolve an Agent, which Branches and Trajectories to run, how to divide the Attempt
budget, and which Kernel or adaptive State to route into the next round. Runtime rejects a Workflow
that exceeds the Challenger cap or does not allocate the exact Attempt budget.

Deployment setting `campaign.max_parallel_attempts` independently bounds the number of Optimizer
Sessions admitted from one Workflow parallel batch. It is a concurrency limit, not a topology.

At Bootstrap, Runtime queries Agate. The returned architecture (for example `sm_120`) is Agent
visible; the canonical GPU alias is sealed separately for Agate scheduling.

## Lineage Seed v1

The spec fixes DSL and the same resource envelope, then selects one source:

- `source_type: "artifacts"`: Agent Artifact digest plus Kernel Artifact digest;
- `source_type: "revisions"`: registered Agent Revision ID plus Kernel Revision ID;
- `source_type: "lineage_baseline"`: a source Lineage's frozen Bootstrap baseline (used by
  ablation control).

Runtime revalidates Agent content and re-evaluates the Kernel under the destination Campaign before
publishing independent `agent-v0`/`v0` roots.

## Ablation Arm v1

`seed-ablation-arm --spec` accepts:

- `creation_key` and `source_lineage_id`;
- `optimizer_attempt_budget` and optional `max_challengers` (default 0);
- `workflow_command`, selecting the complete initial executable organization;
- optional `optimizer_model`.

It creates a separate single-Lineage Campaign; by default no Challenger is generated. The selected
Workflow program owns Branch/Trajectory construction, evolution timing, Kernel routing, and whether
each completed Attempt's output State is reused. Runtime interprets neither reset/retain nor
first-Epoch/evolution-frequency flags. The Evolver model and commit are inherited from the source;
the Optimizer model also inherits unless explicitly overridden. The baseline outcome is reused
without remeasurement.

## Validation rules

- Secret names must be valid environment identifiers; secret values stay outside JSON.
- Git production identities are full lowercase commit SHAs.
- Paths inside Agent protocols are safe relative POSIX paths; deployment paths resolve from config.
- Workspace roots and storage locations must be distinct.
- Production launcher read-only binds cannot expose Runtime storage or sibling Worker roots.
- Immutable Campaign inputs cannot change under the same creation key.
