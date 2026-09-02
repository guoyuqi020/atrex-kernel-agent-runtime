# Configuration Reference

English | [中文](configuration.zh.md)

Runtime accepts strict JSON and rejects unknown fields. Local relative paths are resolved from the
file that contains them. Use checked-in examples as syntactically complete templates; this document
explains ownership and invariants rather than duplicating every numeric value.

## Configuration documents

| Document | Schema | Purpose |
| --- | --- | --- |
| [`runtime.example.json`](../runtime.example.json) | Runtime v1 | Deployment, services, policy, Agent backends, storage, and launcher. |
| `examples/*/campaign.json` | Campaign configuration | Operator, private/public contracts, Core commit, DSL Lineages, models, and Epoch topology. |
| [`lineage-seed.example.json`](../lineage-seed.example.json) | Lineage Seed v1 | Add one independent Lineage from sealed Agent/Kernel content. |
| Ablation Arm spec | Ablation v1 | Create an unevolved control Campaign from a source Lineage's Bootstrap baseline. |

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

### `kernel_agent`

Agent Bundle and public-contract size limits. Optional `base_source` identifies the approved Core
repository, Git executable, fetch/archive bounds, and explicitly allowed submodules. Bootstrap
still supplies a full commit SHA; branches and tags are not persisted identities.

### `gpu_wiki`

Optional query-only service URL, bearer-token environment name, timeout, and request/query/response
bounds. Runtime freezes queries before returning results. There is no feedback, upload, or Outbox
configuration.

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
Attempts use `timeout_seconds`.

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
modes share filesystem settings: bwrap executable, private home and workspace mount, resolver,
extra read-only binds, hidden host paths, and optional read-only reference-project root. The
reference-project tree is available to framework Bootstrap only, not ordinary Attempts.

`sandbox.resources` sets memory, swap, CPU quota, and PID limits. Container mode has no per-Session
resource block because the outer container owns aggregate limits. Both modes preserve the
surrounding network namespace and therefore allow public egress and reachable host services.

## Evidence

`campaign.evidence` bounds trace files/bytes/events, normalized text, and Kernel diffs. Redaction
patterns apply only to normalized projections. Original Session Artifacts remain immutable and may
be materialized into authorized Agent Evidence or retrieved by administration; they are never
uploaded to GPU Wiki.

## Campaign configuration

Required top-level fields:

- `creation_key`, `operator`, and Agate environment selector `hardware_target`;
- private `evaluation_contract`;
- exactly one public input: preferred `shape_train`, legacy `agent_problem`, or neither when Core
  problem generalization is configured;
- full `base_revision.commit`;
- `challenger_count`, `challenger_start_epoch`, `trajectories_per_branch`, and
  `attempts_per_trajectory`;
- non-empty `lineages` map keyed by DSL.

Each Lineage provides optional Optimizer/Evolver model identities, `baseline_kernel`, and
`initial_evidence`. Missing model identity delegates to the configured Backend CLI default.
Optional `problem_generalization_model` applies only when Runtime invokes Core problem
generalization.

At Bootstrap, Runtime queries Agate. The returned architecture (for example `sm_120`) is Agent
visible; the canonical GPU alias is sealed separately for Agate scheduling.

## Lineage Seed v1

The spec fixes DSL and Epoch topology and selects one source:

- `source_type: "artifacts"`: Agent Artifact digest plus Kernel Artifact digest;
- `source_type: "revisions"`: registered Agent Revision ID plus Kernel Revision ID;
- `source_type: "lineage_baseline"`: a source Lineage's frozen Bootstrap baseline (used by
  ablation control).

Runtime revalidates Agent content and re-evaluates the Kernel under the destination Campaign before
publishing independent `agent-v0`/`v0` roots.

## Ablation Arm v1

`seed-ablation-arm --spec` accepts:

- `creation_key` and `source_lineage_id`;
- `attempts_per_trajectory` and optional `trajectories_per_branch`;
- `ephemeral_agent_state` (default true);
- optional `optimizer_model`.

It creates a separate Campaign with one `challenger_count=0` Lineage. When
`ephemeral_agent_state=true`, every Attempt starts with empty adaptive `skills/` and `tools/`;
false retains the source Bootstrap deposit and later serial State, isolating Evolver impact only.

## Validation rules

- Secret names must be valid environment identifiers; secret values stay outside JSON.
- Git production identities are full lowercase commit SHAs.
- Paths inside Agent protocols are safe relative POSIX paths; deployment paths resolve from config.
- Workspace roots and storage locations must be distinct.
- Production launcher read-only binds cannot expose Runtime storage or sibling Worker roots.
- Immutable Campaign inputs cannot change under the same creation key.
