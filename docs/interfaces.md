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
| `seed-ablation-arm` | `--config --spec <file>` | Create a control Lineage with an independent evolution schedule in its own Campaign from a source Lineage's Bootstrap baseline. |
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
- Gateway `503` responses include `error="gateway_unavailable"` and an actionable `detail`
  (at most 8 KiB). Runtime logs retain the chained exception and Attempt/operation/request-digest
  correlation; `gateway.operation_failed` also records the exception type and message. Public
  details omit credentials and use a summary for source-tree errors containing private cases.
  A Runtime `503` does not by itself mean Agate returned HTTP 503, or that the failure is retryable.
- A Gateway `400` with a recognized `operation` includes `request_schema`: the Agent-facing JSON
  Schema generated from the same Pydantic model that rejected the request. Runtime-owned fields
  and `idempotency_key` are omitted. It also includes compact `issues` with Agent-visible field
  paths, stable error codes, and repair messages, without echoing request values. An absent or
  unknown operation instead returns `supported_operations`.
- Core tool commands print one JSON Object for expected failures and exit nonzero. The Object keeps
  Runtime `error`, `detail`, `issues`, `request_schema`, or `supported_operations`, and adds
  `status="error"`, `command`, and `http_status` when applicable; expected failures do not emit a
  Python traceback.
- Local Evaluate file errors in Core and Kernel Design Agent identify the failing path
  in `issues[].path`. Evaluate's local `request_schema` includes the canonical
  `full`/`correctness_only` modes, inline/file parameters, and mutually exclusive forms, plus bounded
  field-specific `recovery` steps. Evaluate accepts optional `candidate_path` and a nested
  `comparison` object requiring `method: "abba"` and `baseline_path`, with `repeats` bounded to
  2–20. A comparison requires `mode: "full"`. Nested errors identify `comparison.method`,
  `comparison.baseline_path`, or `comparison.repeats`; input-file errors identify `input_path` or
  `shapes_path`. Existing Runtime-supplied `issues`, `request_schema`, and `recovery` for these
  operations are preserved rather than replaced by local fallback guidance.
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
| `POST /v1/wiki/query` | Retained Wiki integration endpoint; new managed Agent sessions no longer receive Wiki authority. |

Candidate operations upload a complete Base64 file Bundle. Runtime seals it before execution.
Every key is idempotent: the same key/request replays the committed response; changed content with
the same key returns conflict. `evaluate` creates an exploratory evaluation record but does not by
itself retain a Kernel revision. Runtime keeps the original Agate response and its
`gateway_result_digest` private for evaluation, comparison, and audit. Separately, it seals the
canonical Agent-visible `operation`/`status`/`result` projection as a Result Artifact. The Agent
receives its `result_artifact_digest`; initial execution and later reads expose the same canonical
content and never expose the private Gateway Result identity.

The `evaluate` wire request optionally accepts `mode: "full" | "correctness_only"` (default
`full`), `input_py` (UTF-8 Python input-generator source, at most 128 KiB), and `shapes` (a non-empty
object of Agate Shape records keyed by integer strings). Each Shape record is an object. Overrides
are independent: omitted input source or Shapes are reused from the private Contract. The reference
and trusted evaluation policy remain unchanged. `correctness_only` omits performance measurement
and automatic profiling. Custom or correctness-only calls keep their Kernel Trial and Result
Artifact identities, and their nested `result` records `mode` and `input_scope` (`custom` or
`contract`). They do not satisfy the full trusted-contract evaluation required before
`candidate_ready`; the default `{"operation":"evaluate"}` behavior remains unchanged.

See the [paired input and Shape file example](evaluation.md#custom-input-file-example) for complete
contents and the mapping from `input_kwargs` to `_make_inputs`, its return dictionary to
`Model.forward`, and `init_kwargs` to the Model constructor. HTTP accepts contents, not file paths.

An `evaluate` wire request with `comparison: {method: "abba", repeats: 2}` carries source Bundles in
both `baseline` (A) and `candidate` (B). `comparison.repeats` defaults to 2 (range 2–20), and the
comparison requires `mode: "full"`; `input_py` and `shapes` remain optional.
Runtime validates and seals both sources. Per-side observations are interleaved within one
allocation per Shape batch; `comparison.repeats: 2` produces A, B, B, A, and larger schedules must fit the
allocation budget. ABBA is always exploratory and does not satisfy `candidate_ready` or trigger
retention/promotion. It returns B's `kernel_artifact_digest` and a
`result_artifact_digest`. The response retains `operation: "evaluate"` and marks the comparison in
`result.comparison` with `method: "abba"` and `repeats`. The nested result contains `baseline_kernel_artifact_digest`,
`baseline`/`candidate` summaries, `schedule`, all `measurements`, and A/B `speedup` plus
`improvement_pct`. These results are readable through Trial/Result Artifact queries, not the
ordinary Evaluate history routes. See [Evaluation](evaluation.md#exploratory-abba).

For `env`, Core returns the Agent-safe service result directly. Kernel operations always return
`kernel_artifact_digest` and `result_artifact_digest`; `dev` and `profile` keep those identities
beside the flattened Agent-safe Job result. Its nested `result` uses a numeric opaque
`shape_id`, normalizes Kernel duration to microseconds and common resource aliases, retains safe
profiler counters, and adds `kernel_count`, `total_duration_us`, per-Kernel
`duration_share_pct`, `dominant_kernel`, duration-weighted `weighted_sol_pct`, and
`dominant_bound`. Shape inputs and dimensions remain private.

For multi-file source-tree Lineages, `profile`, `check` and `disassemble` keep these APIs but use
Runtime-owned Agate Dev drivers over the complete sealed tree. Check is a one-case compile/launch
probe, optionally under Compute Sanitizer; it is not a correctness Gate. Diagnostic `passed=false`
is a failure even when delivery completed. See [source-tree diagnostics](source-trees.md#profile-check-and-disassemble)
for NVIDIA tool requirements, arguments, output exports and limits.

Agent references use Result Artifact digests, not Trial IDs. `result_artifact_read` retrieves
one exact observation and returns `kernel_artifact_digest`, `result_artifact_digest`, `operation`,
`status`, and `result`. A Kernel may have multiple distinct Results. Replaying the same invocation
returns the same Result; separate invocations preserve separate Attempt/generation ownership.
The Agent-facing `kernel-trial-show` command is removed. Runtime and Core/KDA Bundles must be
updated together; Trial-based callers must pass a Result Artifact digest instead. Internal Trial
grouping and administrator history routes are unchanged. For example, an Experiment uses
`"before": {"result_artifact_digest":"sha256:<before-result>"}` and
`"after": {"result_artifact_digest":"sha256:<after-result>"}`; Runtime freezes only those selected
observations, not every measurement ever made of either Kernel.
`kernel_artifact_read` accepts the returned `kernel_artifact_digest` (as
`kernel_artifact_digest`), required `file` destination under `scratch/`, and optional
`artifact_file` source path (defaulting to the destination basename). The Core tool atomically
writes the exact bytes and returns only status, path, byte count, and SHA-256. `result_artifact_read` accepts one
Observation's `result_artifact_digest` and reads its normalized Agent-visible Result Artifact.
The returned `operation`, `status`, and `result` match the initial operation. Evaluate views without comparison
contain `operation`, `status`, a correctness verdict plus worst-case `rel_err`, `max_abs_err`, and
`max_rel_err`. Full evaluations additionally report both aggregate latencies and latency by opaque
Shape ID; correctness-only results contain no performance measurements. Custom and correctness-only
views preserve `mode` and `input_scope`. Comparison results use the `result.comparison` marker and A/B
summary described above; private evaluator inputs and hidden-case details remain withheld. These
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

For `gateway-execute` with `operation: "evaluate"`, Core supports `input_path` and `shapes_path`
alongside the inline wire fields. These safe workspace-relative paths name regular UTF-8 files:
Python input source (at most 128 KiB) and a JSON object of Shape records (at most 256 KiB).
Core rejects absolute/traversal paths, symbolic links, `.runtime` control paths, missing or special
files, malformed UTF-8/JSON, and both inline and path forms for the same component. It expands files
before hashing the request, so file-content changes produce a new idempotency key and equivalent
inline contents produce the same key. For example:

```json
{"operation": "evaluate", "mode": "correctness_only", "input_path": "scratch/custom-input.py", "shapes_path": "scratch/custom-shapes.json"}
```

Use `{"operation":"evaluate","mode":"correctness_only"}` to check contract inputs without
timing, or `{"operation":"evaluate"}` for the default full trusted-contract evaluation.

If loading an override fails locally, use its `issues[].path` and `recovery` to repair the named
path, regular-file contents, UTF-8 encoding, Shape JSON object, or inline/path conflict before
retrying. The attached `request_schema` describes the Agent-authored Evaluate request, including
both inline and path forms; it does not expose trusted request fields or private evaluator inputs.

For `operation: "evaluate"`, Core and Kernel Design Agent accept optional `candidate_path`
(default `work/kernel`). Selecting an ABBA comparison additionally requires
`comparison: {method: "abba", baseline_path: "scratch/baseline.py"}`. Both source paths name a workspace-relative `.py` file or
Kernel Bundle directory; a single `.py` file maps to `kernel.py`, and directories preserve relative
file names. Safe path rules and Candidate Bundle limits apply to both sides. The tool uploads
`baseline` and `candidate` automatically; these wire fields are not Agent-authored, and the local
paths are removed before content-based idempotency is computed; the wire `comparison` retains only
`method` and optional `repeats`. The same Evaluate input-file
helpers are supported for a shared custom input generator and Shapes.

```json
{"operation": "evaluate", "candidate_path": "scratch/candidate.py", "comparison": {"method": "abba", "baseline_path": "scratch/baseline.py", "repeats": 2}}
```

| Command | Agent-authored request |
| --- | --- |
| `gateway-execute` | One GPU/Agate operation and its parameters; Candidate operations upload the working Kernel by default. Evaluate can select Candidate B with `candidate_path` and comparison baseline A with `comparison.baseline_path`. |
| `kernel-artifact-read` | Copies exact visible Kernel source by Artifact Digest into a required `scratch/` destination; stdout contains only the write result. |
| `result-artifact-read` | Reads a normalized Agent-visible Result Artifact by digest; request JSON omits `operation`. |
| `update-direction` | Creates an immutable Direction definition with `propose`, or updates an existing Direction with `start`, `complete`, `abandon`, `block`, or `defer` plus analysis. Closures explicitly select supporting Experiments and declare hypothesis_status; returns the stable Direction ID. |
| `list-directions` | Requires a safe `file` under `scratch/`; atomically writes Direction ID, name, lifecycle status, hypothesis_status, and any declared ancestry to that file and returns only status, file, and count. |
| `load-direction` | With exactly one `direction_id`, returns the complete normalized Direction, including hypothesis_status, all associated_experiment_ids and the latest explicitly selected supporting_experiment_ids. |
| `record-experiment` | Records its `direction_id`, before/after Result Artifact digests, factual `evidence`, interpretive `analysis`, and action. Runtime freezes the Trials' Kernel and Result Artifact identities. Every Experiment needs at least one Kernel-bound Gateway Result. `abandon_direction` may be one-sided; Bootstrap `baseline` requires only `after`. Returns the stable Experiment ID. |
| `list-experiments` | Requires a safe `file` under `scratch/`; atomically writes Experiment ID, name, hypothesis, change, evidence, analysis, and action from frozen history plus the current live Journal, then returns only status, file, and count. |
| `load-experiment` | With exactly one `experiment_id`, returns that complete Agent-visible Experiment without Runtime-internal ordering metadata. |
| `attempt-report` | Terminal schema-v12 Agent handoff with engineering evidence, Direction events, and Direction-bound Experiments. Both `framework_baseline` and ordinary optimization use it; Bootstrap may report only `candidate_ready` or `blocked`. It has no duplicate next-direction list or top-level `decision`; Runtime alone decides retention. |

The example configurations and production workspace generator set
`campaign.optimizer.max_attempt_report_bytes` to `1048576` (1 MiB). This limits the complete
terminal Report, including the Journals attached by the tool. Core/KDA check the assembled size
before submission; the Runtime proxy checks it before acceptance, and the worker checks again
when reading the file after the Session. Core/KDA also limit each `--request` JSON file to 1 MiB, counted as file bytes including
whitespace. These are independent limits; the HTTP request-body limit and the custom-input
source/Shape-file limits are unchanged. Existing workspace configs retain their saved value
unless explicitly updated.

### Report completion after normal model exit

`campaign.optimizer.report_completion_retries` defaults to `2` (integer `0..10`).
After a successful model invocation, Core/KDA query Runtime's `attempt_report_status` operation
on `POST /v1/runtime/queries` with the current Attempt capability. This harness-internal query
returns `missing` or `accepted` with the sealed Report; it is unmetered, invokes no Agate Job,
and never treats an Agent-written file or final chat message as acceptance. Accepted Reports are
restored locally if needed, without repeating the write-once submission.

When missing, the harness starts at most that many report-only provider invocations in the same
Attempt and workspace, pointing to the existing Journal, draft and Trace. This is a fresh provider
conversation, not native resume, a new optimization Attempt, or permission to invent measurements.
An unaccepted local terminal file is moved to a unique scratch backup so it cannot block resubmission.
All invocations share the original wall-time deadline and token/credit allowance. Nonzero model exits,
timeouts, exhausted quotas, incomplete provider capture, or unavailable usage do not trigger
completion. A known Claude usage-reconciliation gap with captured, nonzero provider counters
is an accounting warning, not a failed optimization: report completion and normal candidate
validation still run. Other process/policy failures (including other causes of exit 126) remain
blocking.
`0` disables the extra invocations but still checks acceptance.

The Trace retains the initial capture at its root and later captures under `continuations/001/`,
`002/`, etc. Root `session.json` indexes `segments` and `report_completion`;
`conversation.jsonl` combines them with segment identities, and provider usage is cumulative.
After exhausted retries, the Worker Session ends as `report-completion-exhausted`, with no
successful candidate. Runtime classifies that physical Session as an incomplete handoff and starts
a fresh recovery Session for the same logical Attempt or Bootstrap run, subject to the normal
`max_infrastructure_retries` budget. The failed capture and Runtime State checkpoint remain
auditable; the configured Attempt count is unchanged. Only exhaustion of that outer recovery budget
surfaces the failure, and the incomplete handoff is never consumed as an ordinary negative
optimization result. The bounded report-only continuation applies to Core/KDA optimization and
framework baseline, not problem generalization or Evolver; the outer Runtime classification also
protects older Agent commits that simply exit successfully without an accepted Report.

### Terminal handoff and Journal

Every `attempt-report` status, including `blocked` and `pivot`, is rejected with HTTP 409,
`error: gateway_calls_in_progress`, and `pending_operations` while Gateway calls are executing
in the same Attempt and recovery generation. This applies to Bootstrap and optimization alike.
Wait for the already-started local tool commands and read their terminal results, then retry the
report; do not launch duplicate measurements or end a headless Session expecting a later wake-up.
The check and call admission are atomic across Runtime processes sharing Gateway Control SQLite.
Calls remain active through result persistence; success, failure, and cancellation release them.
Local Journal/history reads and report-status queries do not block handoff. A process crash leaves
a fail-closed reservation scoped to its recovery generation; normal Attempt recovery fences it off.

`candidate_ready` requires non-empty matching Runtime-owned Direction and Experiment journals and
evidence-backed Findings. `blocked` and `pivot` may have empty journals and Findings when no
Direction needs closing; give the genuine reason in the report rather than fabricate evidence.
Any in-progress Direction must still be closed with an associated Experiment first.
The first successful `attempt-report`
call publishes a write-once terminal Report. Validation or tool errors publish nothing, so the Agent
may correct the request using `issues`, `request_schema`, and `recovery` and retry; a successful call
must not be repeated.
Every Experiment names a visible Direction that is in progress or closed (`completed`, `abandoned`,
`blocked`, or `deferred`). Late Experiment submissions can attach existing evidence after closure;
they append to the current Attempt's Journal without reopening the Direction, changing its status,
or rewriting prior events. Its associated Experiment IDs update; selected closure support does not. A merely
`proposed` Direction still must be started first. Trial visibility, ownership, and evidence validation
remain unchanged; this is not permission to resume research without `start`.
Before terminal handoff, no Direction may
remain in progress. `complete`, `abandon`, `block`, and `defer` each require at least one Experiment
explicitly selected in `supporting_experiment_ids` (1–32 unique IDs), all visible and belonging to
that Direction. `propose` and `start` do not select support. Every closure also requires
`hypothesis_status=unresolved|supported|refuted`, independently of lifecycle. For supported/refuted,
each selected Experiment must bind a completed Gateway Result in its `after`; Runtime validates
bindings, not causal relevance or scientific truth. Untested interpretations remain unresolved.

Every new Experiment must bind at least one real Kernel-bound Gateway Result. `abandon_direction`
permits before-only or after-only evidence, never both-null. `keep_after`, `restore_before`, and
`adopt` require both sides; Bootstrap `baseline` requires null before and non-null after.
Check/Profile and failed diagnostic Results may document a blocker without a performance claim.
Health/Env and unbound Dev results do not qualify. A transport error without a Result Artifact is
not citable: if no real evidence exists, closure remains blocked and normal session recovery handles
the failure; never fabricate an Experiment to finish. Blocked Bootstrap reports may omit baseline
when all Experiments are diagnostic `abandon_direction` records; candidate_ready still needs baseline.
Read-only replay of Runtime-owned historical records still accepts old unmeasured `block`/`defer`
events without altering them; that compatibility context is not available to Agent requests.
One Attempt may start and advance at
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
terminal journals, Kernel Trials, and Result Artifacts become the root history of ordinary Attempts
in that Lineage.

Use `record-experiment` with `action="adopt"` to select an unchanged Kernel from visible history.
Both `before` and `after` contain exactly `{"result_artifact_digest":"sha256:<result>"}`. Unlike other actions, `adopt` allows a
historical `after`: Runtime requires a successful ordinary full Evaluate of that exact Kernel,
a committed matching Result Artifact, and the same operator, hardware, DSL and sealed evaluation
contract. Existing history visibility still applies, including explicitly inherited Bootstrap
history. Custom inputs, correctness-only checks, Profile and exploratory ABBA do not qualify.
The Experiment records the current decision while preserving the original Trial and measurement
identities; it does not create another measurement or change the historical Trial's disposition.
This persisted adoption can satisfy `candidate_ready` without reevaluating the unchanged candidate.
A different candidate needs its own qualifying evidence; a current failed full Evaluate cannot be
overridden by adopting an earlier success. Request idempotency is scoped to Attempt and recovery
generation, not a global prohibition on evaluating an Artifact in another Attempt.
`list-experiments` and `load-experiment` combine the current live Runtime Journal with prior durable
Journals; terminal Report Artifacts remain a compatibility fallback for older records. Completed Epoch history includes journals from the
selected branch and every losing Active/Challenger branch, while branch, Epoch, Attempt, selection,
and current/history provenance remain hidden from the Agent. Ordinary Agent/Kernel Evidence carries
every completed branch keyed by branch label. In-progress visibility remains
limited to earlier Attempts on the same trajectory; parallel branches become visible only after the
Epoch barrier. These reads use the Attempt-scoped Runtime Journal endpoint, never contact Agate,
consume no Gateway quota, and cannot select an
arbitrary Attempt or Lineage.
Direction history follows the same completed/all-path and in-progress/same-trajectory visibility
boundary. Agent-facing Direction results intentionally hide Branch, Epoch, Attempt, selection, and
current/history provenance.
`load-direction` derives the reverse Experiment association from each visible Experiment's
`direction_id`; recording an Experiment therefore updates the loaded Direction view immediately.
`associated_experiment_ids` is this complete association list. `supporting_experiment_ids` is only
the latest explicit closure selection; late records do not rewrite it. `list-directions` and
`load-direction` expose `hypothesis_status`. Restarting resets the current assessment to unresolved;
immutable prior events remain in history. Old closures lacking an explicit assessment read as
unresolved, with their old automatic IDs treated only as associations. Old unmeasured Experiment
records remain readable, but cannot support new closures. Reports and downstream memory preserve
the explicit assessment and support IDs, which remain Agent interpretations.

For example, after recording a diagnostic Experiment, close without falsely refuting the hypothesis:

```json
{"action":"defer","direction_id":"direction_<id>","analysis":"Diagnostic check completed, but performance hypothesis remains untested","hypothesis_status":"unresolved","supporting_experiment_ids":["experiment_<id>"]}
```

A one-sided diagnostic Experiment (replace placeholders with real IDs):

```json
{"direction_id":"direction_<id>","name":"Compilation check","hypothesis":"The proposed implementation compiles","change":"Attempted implementation","before":null,"after":{"result_artifact_digest":"sha256:<check-result>"},"evidence":"The check returned a compiler diagnostic","analysis":"Compilation is blocked; no performance conclusion","action":"abandon_direction"}
```

Tool validation errors include `issues` (field, code, message), `request_schema`, and `recovery`.
Both-null subjects point to `before` with a message naming both sides. Unknown/cross-Direction
support, duplicate IDs, missing closure fields, and incomplete Gateway evidence are rejected before
journal append. Repair the request using real recorded IDs and retry; an HTTP 400 or validation
failure is not an Experiment and must not be cited as one.

`profile_evidence` is either `null` or an exact object containing `tool_used`, `profiler`,
`profile_level`, `bottleneck_type`, `evidence_summary`, `evidence_chain`, and a non-empty
`supporting_results` array. Each supporting result binds `operation` (`profile` only),
`kernel_artifact_digest` and `result_artifact_digest`. Core/KDA checks the
Runtime-projected `citable_profile_results`; Runtime independently verifies the two Artifact identities
and operation against durable, visible Gateway observations. No Experiment reference is required:
historical Profiles and Profiles obtained after an Experiment snapshot remain citable without
supplementing the Journal or reopening a Direction. Existing history visibility boundaries still
apply. Pending operations without a Result Artifact and non-Profile operations are not citable.
`null` is required when no recorded Profile evidence is available.
Every Finding requires a non-empty unique `supporting_experiment_ids` array. Each ID must name an
Experiment in the same attached Journal, so a Finding resolves through that Experiment's available
before/after subjects to exact Kernel Artifacts, Trials, and Result Artifacts without repeating those
identities in the Finding itself.
`contributing_result_artifact_digests` is a required array naming the historical Result Artifacts whose Kernel code
or approach the Attempt drew content from, and is empty when it drew from none. Core/KDA and Runtime
accept any order and repeated IDs, then sort and deduplicate them before submitting or sealing the
Report. Both validate every supplied ID and enforce at most 64 input entries before deduplication;
neither resolves it against visible history, because the report is an
Agent interpretation rather than a measured fact. Experiment subject identities, by contrast, are
validated against Runtime-owned Trial records. Runtime carries the contributing IDs into the derived Final Report, so later Attempts and the Evolver can read
it.
The Gateway defines no low-level Agate `submit` passthrough and no standalone `sol` operation.
Measurements use Runtime-constructed `evaluate`, optionally with an exploratory comparison; SOL profiling remains available through
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
private Gateway Result Digest is repeated inside the Kernel outcome projections; Experiment
provenance retains the exact Agent-visible Result Artifact Digests. The Runtime-owned
`production_gate` reports
whether content-level production policy was disabled, passed, failed, or not reached. A failed
Gate includes the exact trusted rejection reason.

The Agent handoff schema and sealed Artifact never contain or request a retention ABBA operation.
After a `candidate_ready` handoff is durably recorded, Runtime applies the configured
`kernel_retention_comparison`. When that policy is `same_allocation_abba`, Runtime performs ABBA,
updates the Candidate Kernel Revision with that authoritative Gateway result, and exposes it only
through the Runtime Final Attempt Report. A missing or non-ready handoff terminates without running
the retention comparator.
This authoritative comparison does not create an Agent `gtrial`; never wait for it to fill the
Experiment journal before handoff. Agent-requested ABBA has its own candidate Trial but remains
exploratory and cannot replace the successful ordinary full Evaluate required for nomination.

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

`kernel-artifact-read` copies one Artifact file into `scratch/`; source is not printed:

```json
{"kernel_artifact_digest":"sha256:<kernel>","artifact_file":"kernel.py","file":"scratch/recovered/kernel.py"}
```

```json
{"status":"completed","file":"scratch/recovered/kernel.py","bytes":4281,"sha256":"<file-sha256>"}
```

`result-artifact-read` reads one normalized Agent-visible Result Artifact:

```json
{"result_artifact_digest":"sha256:<result-artifact>"}
```

```json
{"kernel_artifact_digest":"sha256:<kernel>","result_artifact_digest":"sha256:<result>","operation":"evaluate","status":"completed","result":{"correct":true,"correctness":{"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},"latency_us_geomean":12.288,"latency_us_arith_mean":12.400,"latency_us_by_shape":{"0":12.288}}}
```

## Evolver filesystem interface

Evolver has no Runtime Tool or Runtime HTTP capability. Runtime materializes one frozen filesystem
view keyed by Lineage version. `input/agents/agent-vN/` contains one complete Agent Bundle:
implementation, configuration, and `prompts/`, `insights/`, `skills/`, `tools/`.
The writable `candidate/` has the same layout. Existing checkpoints replace packaged defaults;
there is no second Source/State pair to edit.

`input/evidence/agent-vN/` contains an optimization summary and supplementary
`resources/trajectories/<N>/` snapshots. Only participants in the last completed Epoch also have
its Conversations and Attempt reports. Prior reports at `input/evolution-reports/evo-N.json`
link `parent.path` and `generated_agent.path` to complete Bundles; contributing paths refer to
locations in the original producing Session, not guaranteed-current resource contents.

For `evolve_from_history`, copy the selected complete historical Bundle into Candidate before editing.
Declare that revision as `kernel_agent_revision_id` and report the exact sorted Bundle-relative
`changed_paths`, including all four adaptive directories. Runtime revalidates the full diff and seals
the complete Bundle plus a four-directory checkpoint. Optimizer permissions and inheritance rules
are unchanged: implementation is read-only, the four adaptive directories remain writable.

`contributing_paths` records sorted, unique workspace-relative files or directories actually incorporated from
`input/agents/agent-vN/` or `input/evidence/agent-vN/resources/`, including Parent resources from other
Trajectories. Mere reading and automatic Parent inheritance are not contributions. Paths must exist,
contain no links/traversal, and belong to eligible evaluated history or Parent, never a same-Epoch
unevaluated Challenger. `reuse` requires `[]`. Runtime records ownership and exact content snapshots
in the Evolution Trace; the field does not change the Bundle base or revision ancestry.

Evolver submits a draft through its local `evolution-report` tool. Invalid submissions return `issues`,
`request_schema`, and `recovery` without publishing; the first success atomically writes
`scratch/evolution-report.json`. Runtime independently revalidates the report after Session exit.

## Direction genealogy

`update-direction` proposals optionally declare ancestry in addition to their existing definition:

```json
{
  "action": "propose",
  "name": "combine the two measured changes",
  "hypothesis": "disjoint mechanisms can be combined",
  "rationale": "the cited experiments isolated each component",
  "plan": ["port both changes onto the incumbent", "measure the combined Kernel"],
  "success_criteria": "correct and faster under the existing Gate",
  "stop_conditions": "component interaction erases the gain",
  "relationship": "combination",
  "derived_from_direction_ids": [
    "direction_11111111111111111111111111111111",
    "direction_22222222222222222222222222222222"
  ],
  "derived_from_experiment_ids": ["experiment_33333333333333333333333333333333"]
}
```

Relationship types are `retry`, `refinement`, `reimplementation`, `correction`, `port`, and
`combination`. The existing `rationale` explains the connection. Either parent list may be omitted;
each permits at most 32 unique IDs from the caller's existing visible history. Experiments imply their
owning Directions as parents. Every relation needs at least one parent; combinations need two distinct
parents. A correction may set `supersedes_direction_id` to a parent without changing its lifecycle.
Validation precedes persistence. Lifecycle updates cannot rewrite ancestry; propose a new Direction
to correct an earlier declaration. Resume an unchanged unfinished hypothesis with its existing ID.
List/load return declared ancestry; old records remain valid and acquire no fabricated links.

Use `list-directions` and `load-direction` to inspect declared ancestry, and `load-experiment` for
referenced evidence. There is no separate graph export or generated Evolver genealogy file.
Relationships remain Agent-authored claims; measurements and Gate rules are unchanged. Simplified
AKA implements the same validation vocabulary in its Supervisor without introducing Runtime's Pool
scheduler. Semantic classification and causal Pool-benefit analysis remain separate analytical work.

## External service contracts

- Agate is accessed through the published `atrex-gateway-client` SDK. Runtime owns credentials and
  request construction; Workers see only the sanitized Gateway projection.
- GPU Wiki query is `POST /v1/knowledge/query`. The local Wiki implements the same v1 contract.
- Complete schema semantics, Evidence layouts, version labels, and Bundle protocols are in
  [Protocols](protocols.md); every deployment field is in [Configuration](configuration.md).
- Evaluation, Production Gate, comparison, Roofline, and SOL semantics are in
  [Evaluation and Promotion](evaluation.md).
