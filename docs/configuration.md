# Configuration Reference

English | [中文](configuration.zh.md)

The repository exposes two kinds of user-authored control documents:

- [`runtime.example.json`](../runtime.example.json) is the deployment template. Runtime JSON uses
  `schema_version: 1`, rejects unknown fields, and resolves local relative paths from the
  configuration file directory. Copy it to the ignored `runtime.json` and adapt it for the
  deployment.
- Each runnable workflow owns an `examples/<workflow>/campaign.json`, for example
  [`examples/bootstrap/campaign.json`](../examples/bootstrap/campaign.json). Campaign JSON uses
  `schema_version: 3` and is consumed directly by `bootstrap --campaign`.

There is no separate Bootstrap configuration. Bootstrap creates or resumes the immutable Campaign
described by Campaign JSON. Evaluation Contracts, Agent Problems, seed Kernels, and initial
Evidence are referenced Campaign inputs rather than additional control-plane configuration files.
Runtime-generated Results, Manifests, Evidence metadata, traces, and token-usage reports are output
protocols and intentionally have no root-level `*.example.json` configuration template.

## Top-level groups

- `server`: ASGI host and port.
- `storage`: distinct Registry, Gateway-control, Agate-job SQLite files and Artifact root.
- `gateway_proxy`: request/Candidate bounds, signing-key environment name, per-DSL changed-path allowlists, and no-op policy.
- `agate`: URL, authentication mode and credential environment names, HTTP/wait timeouts.
- `kernel_agent`: `max_bundle_files`, `max_bundle_bytes`, `max_entrypoint_bytes`, `max_agent_problem_bytes`, and optional `base_source`.
- `gpu_wiki`: external URL, optional bearer-token environment name, and proxy/query/response limits.
- `campaign`: worker composition and scheduling policy.
- `administration`: bearer-token environment name plus request/event bounds.
- `maintenance`: offline Artifact/workspace retention limits.

`kernel_agent.base_source` declares one approved Core repository, Git executable, fetch/archive bounds, and an exact path-to-URL map for allowed submodules. Bootstrap requests supply the full commit. Local checkout paths are development inputs only; Runtime still imports and seals the tracked tree.

## Campaign group

Campaign paths contain separate roots for Attempt, Evolution, problem-generalization, and lineage-baseline workspaces. Fencing requires `lease_seconds > 2 * heartbeat_seconds`. Deployment configuration does not select DSLs; the separate Campaign definition owns that topology. Gateway operations must be unique and include `evaluate`; capability call quota and lifetime are explicit. `dev` and `wiki_query` operations are audited but do not consume the call quota; benchmark, compile, profiling, and other operations remain metered.

`gate_policy` is the single trusted source of correctness and performance semantics. The checked-in
policy matches Atrex Kernel Agent: Optimizer exploration uses one correctness case, one Eval, and a
100 ms eager measurement budget; retention uses one correctness case and 100 ms; Bootstrap runs two
ordered stages `(1 case, 5 ms)` then `(5 cases, 5 ms)` and accepts only if both pass. The second
stage's latency is the Bootstrap `v0` latency. `warmup_iters` is `10`, tolerances are `0.01/0.05`,
candidate/performance timeouts are `20/120` seconds, and the evaluation Job budget is 600 seconds.

`gate_policy.production_gate` enables the trusted content-level Production Gate. Its model default
is disabled; every checked-in release example enables it explicitly. When enabled, Runtime
rejects a Candidate before Agate execution unless its candidate source is valid Python, contains a
self-authored implementation marker for the Lineage DSL, uses no alternate DSL, PyTorch compute
fallback, `torch.ops`, dynamic external-code loading, prebuilt operator library, or unapproved import.
If present, `solution.json` must name the fixed DSL and only approved dependencies. Ambiguous
third-party implementation dependencies fail closed; they are not delegated to the evolvable Agent.
The same policy is rechecked during authoritative finalization and Artifact-seeded Lineage creation.

Bootstrap replaces all Gate-owned fields in the input Evaluation Contract before sealing it:
sampling, tolerances, timeout, full validation mode, clock policy, canonical Atrex Bench version,
and Gate-owned runner overrides. Input runner overrides cannot supersede this policy. Existing
Campaigns retain their sealed policy; changing `gate_policy` requires a new Campaign identity.
Every optimization Attempt is finalized by `kernel_retention_comparison`: ordinary Evaluate runs A
and B independently for its configured `repeats`, while same-allocation ABBA runs its interleaved
schedule. In either method, the B aggregate becomes the Candidate Kernel's authoritative Evaluation;
there is no separate Attempt-final Eval.

Optional `roofline_builder` declares an approved Atrex Bench repository and full commit, absolute
Git/Python executables, fetch/execution/output bounds, and an optional exact
`sku_by_hardware_target` map. When an Evaluation Contract has no Roofline, Bootstrap executes the
canonical converter from that commit before any Agent Session and seals the validated result. See
[Trusted Roofline construction](roofline-builder.md). If it cannot produce a valid Roofline,
Runtime reports the reason and falls back to an NCU SOL Profile after each correct Eval.

`kernel_retention_comparison` and `agent_promotion_comparison` independently select either ordinary
Evaluate:

```json
{"method":"evaluate","repeats":1,"measurement_uncertainty_us":0.0}
```

Runtime submits `repeats` independent ordinary Eval measurements concurrently for both A and B,
requires every run to pass, compares their arithmetic means, and writes B's aggregate latency and
aggregate Result Artifact into the Candidate Kernel revision.

Each such measurement, plus Optimizer Evaluate, Bootstrap, and Lineage Seed Evaluate, uses the same
fixed trusted executor: four Shapes per Agate Job and no more than four concurrent Shape-batch Jobs.
These limits are Runtime invariants rather than Campaign inputs.

Or exact same-allocation ABBA:

```json
{
  "method": "same_allocation_abba",
  "repeats": 2,
  "minimum_improvement_percent": 0.0,
  "allocation_timeout_seconds": 600,
  "shape_batch_size": 1,
  "max_parallel_shape_batches": 4
}
```

`gate_policy.evaluator` supplies one full Atrex Bench commit for ordinary Eval and ABBA:

```json
{
  "repository": "./third_party/atrex-bench",
  "commit": "FULL_LOWERCASE_COMMIT_SHA",
  "git_executable": "/usr/bin/git",
  "fetch_timeout_seconds": 120,
  "max_archive_bytes": 8388608,
  "max_bundle_files": 128,
  "max_bundle_bytes": 4194304
}
```

The checked-in template points at the repository's pinned `third_party/atrex-bench` Git submodule.
It pins Atrex Bench to a full commit and bounds Git import and the uploaded evaluator-only Bundle.
One Agate `dev` Job owns
the complete interleaved schedule for each Shape batch. Every A and B run must pass; Runtime compares
the geometric mean across repeats and requires a gain strictly greater than
`minimum_improvement_percent`. See [Performance gates](performance-gates.md).

`optimizer` contains the Runtime-authoritative `agent_backend`, `reasoning_effort`, and
`session_settings` binding plus the Core command prefix, explicit/inherited environment,
isolated-home key names, provider-usage report paths, report and diagnostic bounds, timeout/grace,
`max_session_tokens`, and `max_session_credits`. QoderCLI selects the credit quota and records
provider-native credits; Claude, Codex, and Pi select the token quota and record disjoint provider
token buckets. Supported Backend values are `claude`, `codex`, `qodercli`, and `pi`.
`timeout_seconds` limits normal Optimizer Attempts and Problem Generalization;
`bootstrap_timeout_seconds` independently limits each Lineage Bootstrap Session and defaults to
10,800 seconds (180 minutes).
`evolver.timeout_seconds` independently limits one Challenger-building Session; checked-in and
production templates set it to 10,800 seconds (three hours), with a 10-second termination grace.
Runtime applies this deployment binding to Core sessions; the Campaign selects the concrete model
separately. Core still owns prompts, tools, workflow, and Adapter
implementation, while `atrex-agent.json` supplies standalone defaults.

`launcher.backend_credentials` controls host CLI login-state reuse and is enabled by default. For
the selected Backend only, Runtime discovers the configured host Home (or the Runtime process
`HOME`) and projects Claude `.claude`/`.claude.json`, Codex `.codex`, QoderCLI
`.qoder`/`.qodersec`, or Pi `.pi/agent` directly into the Session-specific isolated Home. These
credential mounts are read-only and ephemeral: no credential file is copied into a Workspace
Artifact. Mutable provider state is copied into or created under the private Session Home; in
particular, Codex `config.toml` is writable there so TUI workspace trust can be recorded without
changing the host configuration. CLI caches/session output remain in writable Session paths. User-local CLI installation
roots hidden by the production private Home are restored read-only as well. `host_home` can pin a
deployment-owned source Home; `development_bwrap_executable` defaults to `/usr/bin/bwrap`.
Production Optimizer and Evolver Sessions retain this selected-Backend-only boundary. Interactive
Optimizer/Evolver dev shells are the explicit exception: they project every available supported
CLI login state read-only, give each CLI private writable state under the dev-shell Session Home,
and expose all four CLIs in one shell so an operator can compare or troubleshoot them.

`evolver` contains an independent binding with the same four Backend values, repository, full
commit, Git/import bounds, interpreter command prefix, Bundle/output limits, explicit/inherited
environment, isolated-home keys, trace/token-report paths, timeout/grace, and diagnostics. It has no
token quota. Complete provider usage remains mandatory telemetry for every Backend.
Runtime resolves this environment and imports the pinned Bundle lazily on the first Challenger request.
An Epoch topology with `challenger_count=0` therefore neither requires Evolver credentials nor accesses
the Evolver repository.

`bootstrap_max_parallel_lineages` is a positive Campaign-bootstrap limit and defaults to `1` for
backward-compatible serial bootstrap. Bootstrap seals the shared Evaluation Contract, Agent Problem,
Core commit, and Campaign identity first, then may generate distinct DSL `v0` Lineages concurrently
up to this limit. Result ordering remains the canonical CUDA, Triton, CuteDSL order. Production may
set it to `3` when all three DSL Lineages should bootstrap concurrently.

`max_parallel_branches` is a positive Runtime scheduling limit and defaults to `4`. Evolver
invocations remain serial so later proposals can inspect earlier Challenger designs. After the
Challenger pool is frozen, Active and Challenger Branches run concurrently up to this limit; each
admitted Branch may additionally run all of its configured Trajectories concurrently. Every
Trajectory still serializes its Attempts. A Branch failure is retained independently and does not
cancel sibling Branch tasks; Runtime waits for sibling cleanup before propagating the failure.

`launcher.mode` is mandatory. Both modes keep private evaluator files and paths out of Optimizer,
Baseline, and Evolver workspaces/environments and return only the Agent-safe Gateway result
projection. (The separate problem-generalization phase intentionally receives private inputs.)
`development` retains the explicit clean environment for trusted local debugging. When host CLI
login state is present, Linux bubblewrap supplies a lightweight mount namespace so those entries
and the host root are read-only while the current Workspace and private `/tmp` remain writable.
This does not add cgroup, network, PID, IPC, or Runtime-storage isolation: development mode still
cannot contain a malicious same-user process and is never selected as a fallback. Disable
`backend_credentials` explicitly when running development mode on a platform without bubblewrap
and authenticate through allowlisted environment variables instead.
`container` runs each Worker through bubblewrap inside an operator-supplied outer OCI container.
It requires Linux and bubblewrap, but no systemd, writable cgroup hierarchy, sudo, or per-Session
cgroup. The outer container must allow the user/PID/IPC/UTS namespace operations used by bwrap.
Runtime applies the same read-only root, private `~/workspace`, sibling/Runtime-storage masking,
dropped capabilities, and read-only Backend login-state projection as `sandbox`. Each Worker is
therefore filesystem- and namespace-isolated, while all Workers share the outer container's total
memory/CPU/PID limit. Use a dedicated container, do not mount the Docker socket, Runtime secrets,
private evaluator data, or unrelated host paths into it, and apply resource limits at the OCI layer.

`sandbox` requires Linux, bubblewrap, and systemd with cgroup v2. Its fixed contract is:

- host root read-only, private `/home`, `/tmp`, `/run`, `/dev`, and `/proc`;
- Runtime Artifact/database storage, all four configured Workspace roots, and `hidden_host_paths`
  masked;
- only the current Session bind-mounted read-write at `workspace_mount`, which must be
  `~/workspace` below `sandbox_home`;
- `read_only_bind_paths` available only as explicitly approved immutable dependencies or provider
  configuration; they cannot be located below any Worker root or Runtime storage root;
- per-Session `memory_max_bytes`, `memory_swap_max_bytes`, `cpu_quota_percent`, and `tasks_max`;
- the host network namespace, including the host's DNS, routing, public egress, and reachable host
  services; `resolv_conf` is mounted read-only into the private `/run`.

The cgroup is a system-manager transient service, not a user-manager scope. `systemd_user` is
therefore fixed to `false`, and `worker_user` must name a provisioned non-root account. The trusted
Runtime launcher needs narrowly controlled authority to create that transient service; systemd
then drops to `worker_user` before executing bwrap. The host check also asks systemd to create the
four Worker roots and its probe directly as `worker_user`; this is required on filesystems such as
Lima virtiofs where a successful `chown` may not change ownership. Empty foreign-owned roots can be
recreated, while non-empty mismatches fail closed.

There is no Runtime model proxy or model-host allowlist. Claude, Codex, QoderCLI, and Pi use their
native direct network behavior over the host network. Provider authentication still comes
from the selected Backend's explicitly allowlisted Worker environment or read-only configuration.
This policy intentionally permits arbitrary outbound destinations and does not isolate host
services or Worker-to-Worker traffic. Filesystem and process/resource isolation remain enforced.

`evidence` limits normalized Session summaries and Kernel diffs. Redaction patterns apply only to normalized summaries. Original Session Artifacts remain unmodified and are materialized for Agents or included in bounded Wiki uploads.

## Campaign definition

Only Campaign schema v3 is accepted by the Bootstrap operation. It requires Campaign identity, Evaluation Contract,
`base_revision.commit`, `challenger_count`, `challenger_start_epoch`,
`trajectories_per_branch`, `attempts_per_trajectory`, and per-DSL seed Kernel/initial Evidence.
The keys of `lineages` are the complete initial Bootstrap DSL set; there is no separate `dsls` field
or deployment default. Additional Lineages may later be created from sealed Artifact/Revision roots.
Each `lineages.<dsl>.models` object independently selects optional `optimizer` and `evolver` model
names. Omitted or `null` values use the selected Backend CLI's default model. The Optimizer model is
used for that Lineage's framework baseline and every Attempt; the Evolver model is used for its
Challenger construction. The optional top-level `problem_generalization_model` applies only when
`agent_problem` is omitted. These choices are persisted with Campaign/Lineage state, so resuming an
existing `creation_key` cannot silently change them.
`challenger_count` may be zero; the other topology values must be positive. Epochs before
`challenger_start_epoch` run Active only. Its default value `1` preserves immediate evolution.
Runnable workflows own their concrete `examples/<workflow>/campaign.json` files and may reference
common immutable inputs under `examples/shared`, including a public `agent_problem`; that field may be
omitted only when Core problem generalization is configured. Local `kernel_agent`, precomputed
baseline Gateway result, and baseline latency fields are rejected.

Credential values never belong in JSON. A missing `inherit` variable fails composition;
`inherit_optional` forwards only currently present allowlisted variables and supports mutually
exclusive credential sets such as Claude versus Codex. The capability signing key must be Base64
and decode to at least 32 bytes.

For example, deployment chooses the Backend while each Lineage chooses its models:

```json
{
  "problem_generalization_model": "generalization-model",
  "lineages": {
    "triton": {
      "models": {
        "optimizer": "optimizer-model",
        "evolver": "evolver-model"
      }
    }
  }
}
```

Backend, credentials, executable discovery, reasoning effort, and opaque `session_settings` remain
deployment policy in Runtime JSON. Do not also set a model in Codex or Pi `session_settings` when a
Campaign model is present; the Adapter rejects conflicting model sources.

The separate [`lineage-seed.example.json`](../lineage-seed.example.json) configures an additional
Lineage under an already active Campaign. It does not contain a Commit: it selects Agent/Kernel
content already sealed by Runtime. `source_type` may be `revisions` or `artifacts`; the latter uses
two `sha256:` digests. The new Lineage still inherits operator, hardware target, Evaluation Contract,
and Agent Problem from its destination Campaign.

`campaign.evolver.commit` remains deployment input only until a Campaign is first bootstrapped.
Bootstrap copies that full SHA into the Campaign record. Every resume validates the configured value
against the frozen value; changing Runtime JSON therefore affects new Campaigns only.

`session_settings` is passed only to the selected Backend Adapter. Credentials and CLI locations
remain deployment environment/PATH concerns and are never embedded in this string.
