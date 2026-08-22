# Interface Reference

English | [中文](interfaces.zh.md)

The supported public surface consists of one CLI, three HTTP authorities, Core Runtime Tools, and
frozen Evolver inspection tools. JSON objects reject unknown fields unless explicitly stated.
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
| `show-attempt` | `--config --attempt` | Exact Attempt and disposition. |
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
- Administration: `400` invalid request, `401` missing/invalid token, `404` unknown identity,
  `409` invalid transition. Successful JSON uses `application/json`; Event export is NDJSON.

### Worker routes

| Method and path | Request / response |
| --- | --- |
| `POST /v1/operations` | Gateway protocol v2; executes `evaluate`, `submit`, `profile`, `dev`, `check`, `sol`, `disassemble`, `poll`, `jobs`, `cancel`, `env`, `health`, or `config`. |
| `POST /v1/wiki/query` | Wiki protocol v1 `{schema_version, attempt_id, idempotency_key, query}`; returns a frozen response. |

Candidate operations upload a complete Base64 file Bundle. Runtime seals it before execution.
Every key is idempotent: the same key/request replays the committed response; changed content with
the same key returns conflict. `evaluate` creates an exploratory evaluation record but does not by
itself retain a Kernel revision.

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
| `GET /v1/admin/attempts/{id}` | Attempt detail. |
| `GET /v1/admin/attempts/{id}/worker-sessions` | Attempt Session list. |
| `GET /v1/admin/attempts/{id}/evaluations` | Evaluation list. |
| `GET /v1/admin/attempts/{id}/evaluations/{eval}` | Evaluation detail. |
| `GET .../evaluations/{eval}/{source,result}` | Exact bounded candidate files/raw result. |
| `GET /v1/admin/kernels/{id}` | Kernel detail with measurements. |
| `GET /v1/admin/kernels/{id}/{source,measurements}` | Exact bounded files or measurement list. |
| `GET /v1/admin/agent-revisions/{id}` | Agent revision detail. |
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
python src/runtime_tools.py <command> --request scratch/request.json
```

| Command | Agent-authored request |
| --- | --- |
| `gateway-execute` | One Gateway operation and its operation-specific parameters. Candidate operations automatically upload the current working Kernel tree. |
| `wiki-query` | `query` and optional `idempotency_key`; stdout is only Wiki `content`, not audit metadata. |
| `record-experiment` | Exactly `name,hypothesis,change,evidence,result,decision`; decision is `continue`, `revert`, or `pivot`. |
| `attempt-report` | Terminal schema-v2 report: status/decision, hypothesis, bottleneck, plan, change/profile/evaluation evidence, interpretation, sources, lessons, and next directions. |
| `lineage-bootstrap-report` | Terminal Baseline report with `baseline_ready` or `blocked`; a ready report names positive latency and exact Candidate/Result digests. |

`attempt-report` requires a non-empty contiguous Experiment journal and is write-once. Runtime does
not trust an Agent's success text: it independently reads Gateway records and applies finalization.

## Evolver Runtime Tools

Evolver receives a frozen `runtime-tools/evolver_tools.py` with no HTTP credential. It can read only
the materialized Lineage/Epoch snapshot and mutate only `candidate/` through `candidate-reset`.

| Command | Parameters | Result |
| --- | --- | --- |
| `history` | `--limit` | Completed Epoch Active, Branches, winners, and Kernel versions. |
| `branches` | `--epoch` | Every Branch, Attempt counts, valid/failed/retained candidates, best Kernel. |
| `attempts` | `--epoch --branch`, optional `--trajectory --limit` | Exact Attempt summaries and Evidence paths. |
| `kernels` | optional `--epoch --limit` | Frozen Kernel catalog. |
| `kernel-read` | `--revision`, optional `--file` | File manifest or one UTF-8 source file. |
| `agents` | `--limit` | Frozen Agent catalog. |
| `agent-diff` | `--base --candidate`, optional diff bound | Repository file/status/diff view. |
| `candidate-reset` | `--base <agentrev>` | Atomically replace Candidate with a completed historical Agent and record the base. |
| `trace-paths` | optional `--epoch --limit` | Paths to original, unredacted materialized Session Trace trees. |

Branch names accepted by `attempts` are `active` and `challenger-NNNN`. `candidate-reset` refuses
current-Epoch Challengers and any Agent outside completed Lineage history.

## External service contracts

- Agate is accessed through the published `atrex-gateway-client` SDK. Runtime owns credentials and
  request construction; Workers see only the sanitized Gateway projection.
- GPU Wiki query is `POST /v1/knowledge/query`. The local Wiki implements the same v1 contract.
- Complete schema semantics, Evidence layouts, version labels, and Bundle protocols are in
  [Protocols](protocols.md); every deployment field is in [Configuration](configuration.md).
