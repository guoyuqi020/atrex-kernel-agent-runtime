# Interface Reference

English | [中文](interfaces.zh.md)

The supported public surface consists of one CLI, three HTTP authorities, Core Runtime Tools, and
the Evolver's frozen filesystem input contract. JSON objects reject unknown fields unless explicitly stated.
Typed IDs use stable prefixes such as `campaign_`, `lineage_`, `epoch_`, `attempt_`, `kernelrev_`,
`agentrev_`, and `sha256:`.

The Python module graph is an internal implementation API in release 0.1. The supported embedding
boundary is the CLI or HTTP service; importing `atrex_runtime.*` does not carry compatibility
guarantees unless a symbol is explicitly documented here.

## CLI

All commands use `atrex-kernel-agent-runtime`. Commands that read deployment state require
`--config` except `digest-evolver-bundle`.

| Command | Required selection | Effect |
| --- | --- | --- |
| `serve` | `--config` | Serve health, Gateway, Wiki, and administration HTTP APIs. |
| `bootstrap` | `--config --campaign <file>` | Idempotently create/resume Campaign and initial Lineages. |
| `seed-lineage` | `--config --campaign <id> --spec <file>` | Create a new Lineage from sealed Artifact/Revision roots. |
| `run-campaign` | `--config`, `--campaign <id>` or repeated `--lineage <id>`, `--target-epoch N` | Resume scheduling to an absolute Epoch; optional `--finalize`. |
| `cancel-campaign` | `--config --campaign <id>` | Cancel a quiescent Campaign. |
| `run-task-worker` | `--config` | Claim one durable Task; `--watch` keeps polling. |
| `recover-epoch` | `--config --epoch --recovery-key --reason` | Authorize one idempotent failed-Epoch retry. |
| `dev-shell` | `--config`, `--lineage` or `--attempt` | Enter a real Optimizer workspace without starting Core. |
| `temporary-dev-shell` | `--config --campaign <file>` | Enter a disposable synthetic Optimizer workspace. |
| `evolver-dev-shell` | `--config --lineage --epoch` | Enter a reconstructed frozen Evolution workspace. |
| `temporary-evolver-dev-shell` | `--config --campaign <file>` | Enter a disposable synthetic Evolution workspace. |
| `list-epochs` | `--config`, `--campaign` or `--lineage` | Competition/winner history; `--format json|table`. |
| `list-attempts` | same | All terminal and no-Candidate Attempts. |
| `show-attempt` | `--config --attempt` | Exact Attempt, disposition, input State, and terminal State digests. |
| `list-worker-sessions` | `--config`, one of Campaign/Lineage/Epoch/Attempt/Subject | Model process/Trace catalog. |
| `show-worker-session` | `--config --session` | One Session lifecycle record. |
| `list-kernels` | `--config`, `--campaign` or `--lineage` | Versioned Kernel history; `--format json|table`. |
| `show-kernel` | `--config --kernel` | Kernel, Agent, evaluation, and repeat measurements. |
| `list-agent-revisions` | `--config`, `--campaign` or `--lineage` | `agent-vN` history; `--format json|table`. |
| `show-agent-revision` | `--config --agent-revision` | One Agent revision and provenance. |
| `list-bootstrap-runs` | `--config --attempt` | Every Bootstrap recovery Generation. |
| `show-bootstrap-run` | `--config --attempt --generation N` | One physical Bootstrap execution. |
| `list-evaluations` | `--config --attempt` | Every immutable evaluated Kernel/result pair. |
| `show-evaluation` | `--config --evaluation` | Metadata; `--source` and `--result` add exact bounded payloads. |
| `list-kernel-trials` | `--config --attempt` | Every exact experimental Candidate observed in one Attempt. |
| `show-kernel-trial` | `--config --trial` | Trial operations/decisions; `--source` and `--result` add exact payloads. |
| `gc-artifacts` | `--config --minimum-age-seconds --limit` | Dry-run CAS GC; deletion additionally requires `--apply --confirm-runtime-stopped`. |
| `gc-workspaces` | same | Dry-run Worker-run GC with the same apply confirmation. |
| `digest-evolver-bundle` | `--path` | Validate and digest a Bundle; optional file/byte bounds. |

`--shell zsh|bash` is available on both dev-shell commands. JSON is the stable machine interface;
tables and progress messages are operator presentation.

## HTTP authority and errors

- `GET /healthz` and `GET /readyz` require no authorization.
- `POST /v1/operations` and `POST /v1/wiki/query` require an Attempt-scoped bearer Capability.
- Every `/v1/admin/*` route requires `Authorization: Bearer <admin-token>`.
- Gateway/Wiki: `400` invalid request, `403` invalid/expired/revoked authority, `409` idempotency or
  state conflict, `503` dependency unavailable.
- A Gateway `400` with a recognized `operation` includes `request_schema`: the Agent-facing JSON
  Schema generated from the same Pydantic model that rejected the request. Runtime-owned fields
  and `idempotency_key` are omitted. It also includes compact `issues` with Agent-visible field
  paths, stable error codes, and repair messages, without echoing request values. An absent or
  unknown operation instead returns `supported_operations`.
- Core tool commands print one JSON Object for expected failures and exit nonzero. The Object keeps
  Runtime `error`, `detail`, `issues`, `request_schema`, or `supported_operations`, and adds
  `status="error"`, `command`, and `http_status` when applicable; expected failures do not emit a
  Python traceback.
- Core-owned Trial/Artifact/Result, Wiki, Direction, Experiment, and Attempt Report validators add
  their command-specific JSON Schema. Visibility or lifecycle errors additionally provide bounded
  `recovery` steps that name safe list/load calls or explain which previously returned identity to
  reuse. Recovery never enumerates an inaccessible Lineage identity.
- Agate rejection before Job creation is classified as Candidate/source validation. Safe validation
  details are returned after recursively removing evaluator inputs, references, Shapes, payloads,
  and logs. Failures after hidden-case execution remain redacted.
- Administration: `400` invalid request, `401` missing/invalid token, `404` unknown identity,
  `409` invalid transition. Successful JSON uses `application/json`; Event export is NDJSON.

### Worker routes

| Method and path | Request / response |
| --- | --- |
| `POST /v1/operations` | Gateway protocol v2; executes GPU/Agate operations only. |
| `POST /v1/runtime/queries` | Gateway protocol v2 envelope for unmetered Runtime-local history and source queries. |
| `POST /v1/runtime/journals` | Gateway protocol v2 envelope for unmetered, Runtime-owned Direction/Experiment mutations and reads. |
| `POST /v1/wiki/query` | Wiki protocol v1 `{schema_version, attempt_id, idempotency_key, query}`; returns a frozen response. |

Candidate operations upload a complete Base64 file Bundle. Runtime seals it before execution.
Every key is idempotent: the same key/request replays the committed response; changed content with
the same key returns conflict. `evaluate` creates an exploratory evaluation record but does not by
itself retain a Kernel revision. The wire response retains `schema_version` for validation and
persistence. Before printing the response to the Agent, the Core tool removes that top-level field
and merges the authoritative `evaluation.correct` and `evaluation.latency_us` into `result` as
`correct` and `latency_us`; it removes the top-level `evaluation` and equivalent result aliases.
For `dev`, `disassemble`, `jobs`, `poll`, `cancel`, `env`, `health`, and `config`, Core returns the
Agent-safe `result` object directly. `profile` additionally returns the Kernel Artifact, Kernel Trial, and Gateway Result
identities beside the flattened Agent-safe Job result. Its nested `result` uses a numeric opaque
`shape_id`, normalizes Kernel duration to microseconds and common resource aliases, retains safe
profiler counters, and adds `kernel_count`, `total_duration_us`, per-Kernel
`duration_share_pct`, `dominant_kernel`, duration-weighted `weighted_sol_pct`, and
`dominant_bound`. Shape inputs and dimensions remain private.

`kernel_trial_show` retrieves one known experimental Candidate provenance record by the
`kernel_trial_id` returned by a Gateway response or retained Experiment record.
`kernel_artifact_read` accepts the Trial's `kernel_artifact_digest` (as
`kernel_artifact_digest`), required `file` destination under `scratch/`, and optional
`artifact_file` source path (defaulting to the destination basename). The Core tool atomically
writes the exact bytes and returns only status, path, byte count, and SHA-256. `gateway_result_read` accepts one
Observation's `gateway_result_digest` and returns a normalized Agent-visible view. Evaluate views
contain `operation`, `status`, a correctness verdict plus worst-case `rel_err`, `max_abs_err`, and
`max_rel_err`, both aggregate latencies, and
latency by opaque Shape ID; private evaluator inputs and hidden-case details remain withheld. These
operations are unmetered, never call Agate, and do not accept a caller-selected Lineage or Attempt.
Current-Attempt identities remain available from the original operation response and retained
Experiment records.

### Administration routes

| Method and path | Purpose |
| --- | --- |
| `POST /v1/admin/campaigns/bootstrap` | Bootstrap from Campaign schema v3; HTTP file paths must be absolute. |
| `GET /v1/admin/campaigns/{id}` | Campaign state and frozen provenance. |
| `POST /v1/admin/campaigns/{id}/lineages` | Seed a Lineage from schema v1. |
| `POST /v1/admin/campaigns/{id}/cancel` | Cancel a quiescent Campaign. |
| `GET /v1/admin/campaigns/{id}/{epochs,attempts,kernels,agent-revisions,worker-sessions}` | Campaign-scoped catalogs. |
| `GET /v1/admin/lineages/{id}/{epochs,attempts,kernels,agent-revisions,worker-sessions}` | Lineage-scoped catalogs. |
| `GET /v1/admin/bootstrap-attempts/{id}/runs[/N]` | Bootstrap Generation list/detail. |
| `GET /v1/admin/attempts/{id}` | Attempt detail, including input and terminal Runtime State digests. |
| `GET /v1/admin/attempts/{id}/report` | Runtime Final Attempt Report, fusing the Agent handoff with authoritative parent/Candidate Gateway results. |
| `GET /v1/admin/attempts/{id}/worker-sessions` | Attempt Session list. |
| `GET /v1/admin/attempts/{id}/evaluations` | Evaluation list. |
| `GET /v1/admin/attempts/{id}/evaluations/{eval}` | Evaluation detail. |
| `GET .../evaluations/{eval}/{source,result}` | Exact bounded candidate files/raw result. |
| `GET /v1/admin/attempts/{id}/kernel-trials` | Experimental Candidates, including reverted snapshots. |
| `GET /v1/admin/attempts/{id}/kernel-trials/{trial}` | Trial observations and decisions. |
| `GET .../kernel-trials/{trial}/source` | Exact unversioned Candidate files. |
| `GET .../kernel-trials/{trial}/results` | Exact retained operation-result payloads. |
| `GET /v1/admin/kernels/{id}` | Kernel detail with measurements. |
| `GET /v1/admin/kernels/{id}/{source,measurements}` | Exact bounded files or measurement list. |
| `GET /v1/admin/agent-revisions/{id}` | Agent revision, including Source and Runtime State digests. |
| `GET /v1/admin/worker-sessions/{id}` | Worker Session detail. |
| `GET /v1/admin/epochs/{id}/worker-sessions` | Epoch Session list. |
| `POST /v1/admin/epochs/{id}/recover` | `{schema_version:1,recovery_key,reason}`. |
| `POST /v1/admin/tasks` | Enqueue `{schema_version:1,creation_key,campaign_id,target_epoch_number,finalize}`. |
| `GET /v1/admin/tasks/{id}` | Task state. |
| `POST /v1/admin/tasks/{id}/{cancel,requeue}` | Task lifecycle mutation. |
| `GET /v1/admin/events` | Paginated Events. Query: `after`, `limit`, repeated `kind`, and correlation IDs. |
| `GET /v1/admin/events/export` | Larger bounded NDJSON export with the same filters. |
| `POST /v1/admin/events/prune` | `{schema_version:1,before_sequence,limit}` acknowledged-prefix pruning. |
| `GET /v1/admin/metrics` | Event and Task counters. |

## Optimizer/Core Runtime Tools

Core invokes its bundled `src/runtime_tools.py`. Each request is a JSON object stored under
`scratch/`; `--request` cannot escape that directory. Runtime-owned Attempt IDs, capabilities, and
candidate files are injected by the tool.

```bash
python3 src/runtime_tools.py <command> --request scratch/request.json
```

| Command | Agent-authored request |
| --- | --- |
| `gateway-execute` | One GPU/Agate operation and its operation-specific parameters; Candidate operations upload the working Kernel tree. |
| `kernel-trial-show` | Reads one visible Trial's Kernel Artifact Digest and normalized Gateway results; request JSON omits `operation`. |
| `kernel-artifact-read` | Copies exact visible Kernel source by Artifact Digest into a required `scratch/` destination; stdout contains only the write result. |
| `gateway-result-read` | Reads a normalized Agent-visible measurement for a Gateway Result Digest; request JSON omits `operation`. |
| `wiki-query` | `query`; Core assigns request identity, and stdout is only Wiki `content`, not audit metadata. |
| `update-direction` | Creates an immutable Direction definition with `propose`, or updates an existing Direction with `start`, `complete`, `abandon`, `block`, or `defer` plus analysis. Experiment associations are derived automatically; returns the stable Direction ID. |
| `list-directions` | Requires a safe `file` under `scratch/`; atomically writes Direction ID, name, and current status to that file and returns only status, file, and count. |
| `load-direction` | With exactly one `direction_id`, returns the complete normalized Direction. Its supporting IDs automatically include every visible Experiment bound to it and associations snapshotted internally by status events. |
| `record-experiment` | Records its `direction_id`, measured Kernel/Trial/Result identities, factual `evidence`, interpretive `analysis`, and action. Ordinary comparisons require complete `before`/`after`; Bootstrap alone may use `baseline` with `before=null` and complete `after`. Returns the stable Experiment ID. |
| `list-experiments` | Requires a safe `file` under `scratch/`; atomically writes Experiment ID, sequence, name, and action from frozen history plus the current live Journal, then returns only status, file, and count. |
| `load-experiment` | With exactly one `experiment_id`, returns that complete original Experiment. |
| `attempt-report` | Terminal schema-v12 Agent handoff with engineering evidence, Direction events, and Direction-bound Experiments. Both `framework_baseline` and ordinary optimization use it; Bootstrap may report only `candidate_ready` or `blocked`. It has no duplicate next-direction list or top-level `decision`; Runtime alone decides retention. |

`attempt-report` requires non-empty matching Runtime-owned Direction and Experiment journals. Its first successful
call publishes a write-once terminal Report. Validation or tool errors publish nothing, so the Agent
may correct the request using `issues`, `request_schema`, and `recovery` and retry; a successful call
must not be repeated.
Every Experiment names a visible in-progress Direction. Before terminal handoff, no Direction may
remain in progress, including a started Direction with no Experiment. Such a Direction can be
deferred or blocked; completed and abandoned Directions require supporting Experiments. One Attempt may start and advance at
most three distinct Directions, including inherited and newly proposed Directions. Proposals do not
consume this limit, and the report does not limit how many Directions remain `proposed` or
`deferred`. Only one Direction may be `in_progress` at a time. Starting another is rejected
atomically with `direction_concurrency_conflict`, the conflicting Direction IDs, and recovery steps;
the requested Direction remains unchanged. Their normalized status is the sole next-direction source. Runtime does
not trust an Agent's success text: it independently reads Gateway records and applies finalization.
`update-direction` and `record-experiment` are synchronous Runtime mutations: Runtime validates and
durably appends each event before returning its stable ID. The authoritative Journal is scoped to
the logical Attempt rather than a physical Session or recovery generation. There are no
`scratch/directions.json` or `scratch/experiments.json` authority files. The list/load tools query
the live Runtime Journal merged with authorized frozen history; only their requested compact index
files are written under `scratch/`. A Bootstrap Session starts without prior journal history; after it succeeds, its
terminal journals, Kernel Trials, and Gateway Results become the root history of ordinary Attempts
in that Lineage.
`list-experiments` and `load-experiment` combine the current live Runtime Journal with prior durable
Journals; terminal Report Artifacts remain a compatibility fallback for older records. Completed Epoch history includes journals from the
selected branch and every losing Active/Challenger branch, while branch, Epoch, Attempt, selection,
and current/history provenance remain hidden from the Agent. Ordinary Agent/Kernel Evidence remains
promoted-lineage-only. In-progress visibility remains
limited to earlier Attempts on the same trajectory; parallel branches become visible only after the
Epoch barrier. These reads use the Attempt-scoped Runtime Journal endpoint, never contact Agate,
consume no Gateway quota, and cannot select an
arbitrary Attempt or Lineage.
Direction history follows the same completed/all-path and in-progress/same-trajectory visibility
boundary. Agent-facing Direction results intentionally hide Branch, Epoch, Attempt, selection, and
current/history provenance.
`load-direction` derives the reverse Experiment association from each visible Experiment's
`direction_id`; recording an Experiment therefore updates the loaded Direction view immediately.
`update-direction` snapshots those derived IDs into internal status events; the Agent does not
provide them. Loaded live and snapshotted associations are merged into one de-duplicated list.
`profile_evidence` is either `null` or an exact object containing `tool_used`, `profiler`,
`profile_level`, `bottleneck_type`, `evidence_summary`, `evidence_chain`, and a non-empty
`supporting_results` array. Each supporting result binds `operation` (`profile` or `dev`),
`kernel_artifact_digest`, `kernel_trial_id`, and `gateway_result_digest`. At least one item must be a
Profile result. Core requires every binding to appear in the Attempt's Experiment Journal; Runtime
then verifies that the declared operation and three identities match one durable visible Gateway
observation. `null` is required when no Profile was executed.
Every Finding requires a non-empty unique `supporting_experiment_ids` array. Each ID must name an
Experiment in the same attached Journal, so a Finding resolves through that Experiment's available
before/after subjects to exact Kernel Artifacts, Trials, and Gateway Results without repeating those
identities in the Finding itself.
The Gateway defines no low-level Agate `submit` passthrough and no standalone `sol` operation.
Evaluation uses only Runtime-constructed `evaluate`; SOL profiling remains available through
`profile` with `level="sol"`.

The sealed schema-v12 value is the Agent handoff, not the authoritative outcome. Runtime derives a
schema-v1 Final Attempt Report for the administration route and later Evidence snapshots. It keeps
the engineering narrative, then adds exact `parent_kernel` and `candidate_kernel` objects. Kernel
identity uses `kernel_artifact_digest`, not an internal Revision ID. Each Kernel contains a
normalized `gateway_result` with operation, completion status, correctness, geometric and
arithmetic aggregate latency, and latency keyed by opaque Shape ID. Correctness includes `status`
plus the worst safe aggregate relative-L2, elementwise absolute, and elementwise relative errors;
it never exposes the hidden Shape or Case that produced them. The Candidate additionally
contains its Runtime retention status and aggregate/per-Shape comparison with the parent. No
Gateway Result Digest is repeated inside the Kernel outcome projections; Experiment provenance
continues to retain its exact Gateway Result Digests. The Runtime-owned `production_gate` reports
whether content-level production policy was disabled, passed, failed, or not reached. A failed
Gate includes the exact trusted rejection reason.

The Agent handoff schema and sealed Artifact never contain or request a retention ABBA operation.
After a `candidate_ready` handoff is durably recorded, Runtime applies the configured
`kernel_retention_comparison`. When that policy is `same_allocation_abba`, Runtime performs ABBA,
updates the Candidate Kernel Revision with that authoritative Gateway result, and exposes it only
through the Runtime Final Attempt Report. A missing or non-ready handoff terminates without running
the retention comparator.

```json
{
  "schema_version": 1,
  "attempt_id": "attempt_<id>",
  "status": "candidate_ready",
  "parent_kernel": {
    "version": "v2",
    "kernel_artifact_digest": "sha256:<parent>",
    "gateway_result": {
      "operation": "evaluate",
      "status": "completed",
      "correct": true,
      "correctness": {"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},
      "latency_us_geomean": 200.0,
      "latency_us_arith_mean": 205.0,
      "latency_us_by_shape": {"0": 120.0, "1": 290.0}
    }
  },
  "candidate_kernel": {
    "version": "v3",
    "kernel_artifact_digest": "sha256:<candidate>",
    "status": "retained",
    "gateway_result": {
      "operation": "same_allocation_abba",
      "status": "completed",
      "correct": true,
      "correctness": {"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},
      "latency_us_geomean": 173.28,
      "latency_us_arith_mean": 180.0,
      "latency_us_by_shape": {"0": 100.0, "1": 260.0}
    },
    "comparison_with_parent": {
      "latency_us_geomean_delta": -26.72,
      "improvement_percent": 13.36,
      "latency_us_delta_by_shape": {"0": -20.0, "1": -30.0},
      "improvement_percent_by_shape": {"0": 16.667, "1": 10.345}
    }
  },
  "production_gate": {
    "enabled": true,
    "result": "PASS",
    "failure_reason": null
  }
}
```

### Known Kernel evidence tool examples

The JSON below is the content of the file passed with `--request`. Digests and IDs are abbreviated
only for readability.

`kernel-trial-show` returns only the Kernel identity and normalized Gateway results:

```json
{"kernel_trial_id":"gtrial_<id>"}
```

```json
{"kernel_artifact_digest":"sha256:<kernel>","gateway_results":[{"operation":"evaluate","status":"completed","result":{"correct":true,"correctness":{"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},"latency_us_geomean":12.288,"latency_us_arith_mean":12.400,"latency_us_by_shape":{"0":12.288}}}]}
```

`kernel-artifact-read` copies one Artifact file into `scratch/`; source is not printed:

```json
{"kernel_artifact_digest":"sha256:<kernel>","artifact_file":"kernel.py","file":"scratch/recovered/kernel.py"}
```

```json
{"status":"completed","file":"scratch/recovered/kernel.py","bytes":4281,"sha256":"<file-sha256>"}
```

`gateway-result-read` reads one normalized Agent-visible result:

```json
{"gateway_result_digest":"sha256:<result>"}
```

```json
{"operation":"evaluate","status":"completed","result":{"correct":true,"correctness":{"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},"latency_us_geomean":12.288,"latency_us_arith_mean":12.400,"latency_us_by_shape":{"0":12.288}}}
```

## Evolver filesystem interface

Evolver has no Runtime Tool or Runtime HTTP capability. Runtime materializes one frozen filesystem
view containing current participant repositories and runtime state under `input/agents/`, their
latest-completed-Epoch summaries and Conversations under `input/evidence/`, and completed non-current
Agent versions under `input/historical/agent-vN/`. Prior Agent-creation reports are ordered as
`input/evolution-reports/evo-N.json` and link Source Base/produced-Agent paths. The only writable Agent components are
`candidate/source/` and `candidate/runtime-state/`.

`runtime-state/trajectories/<N>/{skills,tools}/` is the only adaptive Skill/Tool storage. It is
non-versioned state accumulated by Optimizer Sessions and scoped to one Agent revision and
Trajectory. Top-level `skills/` and `tools/` are reserved and rejected in versioned Agent source.
Evolver reads frozen state and may curate one flat Candidate seed at
`candidate/runtime-state/{skills,tools}/`, or revise the versioned Prompt, Workflow, memory policy,
or implementation that governs future use. Runtime seals Candidate source and state independently.
Every new Revision records both Digests as one logical Bundle and copies that exact State into every
new Trajectory.

For `evolve_from_history`, Evolver replaces Candidate Source with a writable copy of the selected
historical `source/`, optionally synthesizes a common state seed from visible historical state, and
declares that Agent Revision as `kernel_agent_revision_id`. Its Source is the proposal reference;
Runtime State identity remains private Runtime control data. Runtime validates Source eligibility,
the reported Source-root-relative Diff, and the private State Diff; there is no Candidate Base side
record or reset command.

Evolver submits its terminal draft with the Bundle-local `evolution-report` command. Validation
errors return `issues`, `request_schema`, and `recovery`; failures publish nothing and may be retried.
The first success publishes `scratch/evolution-report.json` atomically and returns a compact receipt.

## External service contracts

- Agate is accessed through the published `atrex-gateway-client` SDK. Runtime owns credentials and
  request construction; Workers see only the sanitized Gateway projection.
- GPU Wiki query is `POST /v1/knowledge/query`. The local Wiki implements the same v1 contract.
- Complete schema semantics, Evidence layouts, version labels, and Bundle protocols are in
  [Protocols](protocols.md); every deployment field is in [Configuration](configuration.md).
