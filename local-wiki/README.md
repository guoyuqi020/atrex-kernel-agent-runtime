# Atrex Local GPU Wiki

English | [中文](README.zh.md)

This directory is a local HTTP adapter for the independent Atrex GPU Wiki. It is for Runtime
integration tests only. Query behavior is executed by the corpus's own implementation.

The adapter does not implement its own retrieval algorithm. For every query it executes the
selected corpus's `tools/query_nl.py`, including its bridge Agent, intent
validation, operator aliases and component lanes, safe widening, `kernel_wiki` ranking, `hardware_wiki` exact lookup,
and served-record projection. Query `content` is therefore:

```json
{"query_id":"wiki-query-0123456789abcdef0123456789abcdef","records":{"stable.record.id":{"store":"gpu_wiki","wiki_id":"gpu_wiki::stable.record.id","source":"kernel_wiki","type":"technique-card","applies_to":{},"match":{},"payload":{}}},"notes":[]}
```

The complete query result passes through verbatim, including attribution IDs and all notes.
With the public corpus, an unavailable legacy private `internal_gpu_wiki` store is reported by
upstream without preventing public queries. A standalone indexed internal corpus must be selected
directly using the configuration below; it cannot be loaded through that legacy sibling slot.

Runtime continues to provide the versioned HTTP envelope, digest verification, Attempt authority,
and freezing. Every `records` mapping value is already the complete safe
served Record. Upstream provides `query_id` and canonical `wiki_id` for attribution;
there is no separate read operation.

Runtime treats the Wiki as a read-only external knowledge source.

## HTTP interface

| Method and path | Result |
| --- | --- |
| `GET /` or `GET /ui` | Local browser query client. |
| `GET /healthz` | Process liveness. |
| `GET /readyz` | Selected corpus tools, indexes/data dependencies, and SQLite readiness. |
| `POST /v1/knowledge/query` | Strict Runtime query; `content` is upstream `query_id/records/notes`. |

## Corpus

`corpus/gpu-wiki` is ordinary content of this repository, so a checkout is immediately runnable.
Startup copies it into the ignored writable `state/gpu-wiki` store, which is what lets the corpus
tools record query feedback without modifying tracked files. Editing the corpus causes one re-copy
on the next start.

The source commit and copy boundary are recorded in [corpus/README.md](corpus/README.md).
Its original Apache-2.0 license and NOTICE are preserved beside it.

### Internal indexed corpus

The adapter also accepts the standalone internal Wiki's `query_nl.py → query.py → search_index`
layout. Its `query_id/records/notes` envelope is compatible with Runtime. Record contents, including
nested `wiki_identity`, governance metadata, evidence limitations and generation-reference match
labels, pass through unchanged. The adapter does not convert these records to the public schema or
merge the two corpora. `reference_root` selects one upstream implementation per service.

Use [configs/internal.example.json](configs/internal.example.json) to serve the internal corpus on
the same local port as the public example. Import an authorized checkout first:

```bash
# Run from the Runtime repository root; the destination must be new/empty.
mkdir -p local-wiki/corpus/internal_gpu_wiki
set -o pipefail
git -C /path/to/atrex-kernel-agent-internal archive \
  2076c865cc6618d810cfe2bf09b4fc395536693e:internal_source/gpu-wiki \
  | tar -x -C local-wiki/corpus/internal_gpu_wiki
PYTHONPATH=local-wiki/src .venv/bin/python -m atrex_local_wiki serve \
  --config local-wiki/configs/internal.example.json
```

This pin is from `codex/ppu15-agent-wiki`; provenance and the import boundary are in
[corpus/README.md](corpus/README.md). The internal snapshot and its writable state are Git-ignored.
To update it, stop the service, export into a new empty directory and replace the reference tree;
do not overlay archives, which can leave deleted upstream files behind. Startup then refreshes
the writable store. Readiness rejects incomplete indexed stores instead of silently using legacy
retrieval. Snapshot revisions cover native tools, declared shards and governance/evidence inputs;
unchanged file hashes are cached to avoid rereading the full index on each request.
Readiness checks layout, manifest format and dependency presence; validation of shard contents and
governance eligibility remains upstream-owned. A malformed/stale governance projection can still
cause native retrieval to hide records even when all dependency files exist.

Runtime sends the complete hardware/DSL/operator context and original question to the native
front door. Arbitrary HTTP queries use its Claude/Qoder intent bridge; they do not match the
upstream fixed-sentence model-free shortcut. At this internal pin, Claude uses `--bare` and the
bridge environment does not forward `ANTHROPIC_BASE_URL` or `ANTHROPIC_MODEL`. Validate the chosen
bridge CLI/provider separately before deployment; copying local Claude settings alone is not a
verified custom-provider setup. Deterministic native retrieval and a fake intent CLI can test the
index and HTTP contract without contacting a model.

For a separate Claude bridge model, install [scripts/claude](scripts/claude) as `claude` in a
Wiki-only executable directory. Place a `claude-config.json` beside it using
[configs/claude-bridge.example.json](configs/claude-bridge.example.json), and prepend that directory
to **only the Wiki service's** `PATH`. Set absolute `real_claude` and `settings_file` paths and the
authorized `base_url`. The wrapper reads authentication from that settings file without changing
it, and requires its endpoint to match `base_url` before forwarding credentials.

The example selects `model: "qwen3.8-flash"` and `effort: "high"`. These override the native bridge's
model selection and `--effort low`, including the model environment aliases. The Optimizer's
shared settings and model stay unchanged. Keep the executable and its configuration private to
the deployment, and verify a real Wiki query after configuring them. No credential belongs in
the example or wrapper.

### Preloaded indexed queries

`indexed_execution` selects how the indexed corpus runs:

| Value | Behavior |
| --- | --- |
| `"subprocess"` (default) | Starts the native query processes for each request. |
| `"preloaded"` | Loads and prepares the supported internal index at service startup, then reuses that snapshot in isolated query processes. |

[configs/internal.example.json](configs/internal.example.json) enables `"preloaded"`. This mode
requires POSIX process forking and the verified internal tool pin imported above. Unsupported
tool versions fail explicitly; the service does not silently fall back to another execution mode.
Keep the default `"subprocess"` for other corpus implementations.

Preloading prepares the parsed shards, record identities, governance bindings, vocabulary and
operator resolver once per Store revision. A dedicated single-threaded fork server holds the
snapshot; each request forks a child that inherits it and executes the native query implementation.
Query-local environment and stdout remain isolated, while independent children can wait on their
model calls concurrently. `max_concurrent_queries` still bounds active queries. Native ranking,
governance eligibility, widening and result projection retain their original behavior. Responses
are not cached: every request has its own `query_id`, and natural-language queries still run the
upstream intent bridge.

The adapter checks the Store revision before and after each query. A changed revision causes a
new snapshot generation to be preloaded; a Store change during a query still rejects that result.
Preloading adds startup time and memory for the resident parent snapshot in each service worker.
Query children share unchanged snapshot pages through process copy-on-write, with additional
memory for their own work. Size the service and concurrency limit with this memory cost in mind.

For an indexed-query timing breakdown, set `ATREX_WIKI_METRICS_LOG` in the service environment to
a writable JSONL file outside `corpus/`. The native front door records total, bridge and retrieval
latency, bridge attempts and available model token counters. This separates time spent waiting
on the model from local retrieval; preloading does not remove model latency. Set the variable
before starting the service so its query processes inherit it.

This config connects the knowledge service only. In the separate Runtime deployment config,
`gpu_wiki.enabled` defaults to `false`; set it to `true` to expose `wiki-query`, its conditional
instructions and Attempt-scoped authority to Bootstrap/Optimizer. It is independent of the
Direction/Experiment modules. Use Core/KDA source containing the optional Wiki implementation,
then restart Runtime/campaign workers for newly created workspaces; existing sealed Agent source is not
replaced by a config toggle. See [Runtime configuration](../docs/configuration.md#gpu_wiki).

## Run

The checked-in config does not override upstream query defaults. Therefore the corpus's
`query_nl.py` selects its own default bridge CLI, timeout, and Record cap. Optional `agent_cli`,
`query_timeout_seconds`, and `max_results` fields are explicit HTTP deployment overrides; no model
credential is stored by local-wiki.

The current bridge supports `claude` (default) and `qodercli`, using their no-tools JSON
protocols. It does not support `codex`; this restriction applies only to the Wiki intent bridge,
not to Optimizer/Evolver backends. Runtime context and the Agent's question are sent as prose;
intent extraction and operator resolution are entirely upstream-owned. The old local
`operator_families` override has been removed.

`max_concurrent_queries` bounds simultaneous native queries and defaults to `16` in both execution modes.
Additional requests wait for a slot. This prevents unbounded model/subprocess fan-out without
serializing unrelated read-only queries behind one global lock.

The pinned Wiki supports optional query evidence. Set `ATREX_WIKI_PROFILE_ROOT` in the service
environment to a writable run-data directory to save its immutable query events; use
`ATREX_WIKI_TASK_ID` for task attribution. Events include the request, normalized intent, returned
record IDs/ranks, timing and token counters, but not record payloads or coding-agent transcripts.
Without the profile-root variable, no query-event files are created. Trace write failures do not
change the query response. Keep this output outside `corpus/`.

Upstream's AKA plugin and restart-handoff orchestrator are not part of the standalone HTTP service.
The adapter calls `query_nl.py` directly, and ordinary bridge launches need no AKA orchestrator.
Do not pass AKA's `ATREX_ENVIRONMENT_RESTART_HANDOFF_ID` into this standalone deployment.

```bash
PYTHONPATH=local-wiki/src \
  .venv/bin/python -m atrex_local_wiki serve \
  --config local-wiki/configs/local.example.json
```

Open [http://127.0.0.1:8091/](http://127.0.0.1:8091/). When overriding `agent_cli`, use a backend
accepted by the corpus's `tools/agent_launch.py`.

## Verify

```bash
PYTHONPATH=src:local-wiki/src .venv/bin/pytest local-wiki/tests
PYTHONPATH=local-wiki/src \
  .venv/bin/ruff check local-wiki/src local-wiki/tests
PYTHONPATH=local-wiki/src \
  .venv/bin/mypy --config-file local-wiki/pyproject.toml \
  local-wiki/src
```
