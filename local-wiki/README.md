# Atrex Local GPU Wiki

English | [中文](README.zh.md)

This directory is a local HTTP adapter for the independent Atrex GPU Wiki. It is for Runtime
integration tests only. Query behavior is executed by the corpus's own implementation.

The adapter does not implement its own retrieval algorithm. For every query it executes the
corpus's `gpu-wiki/tools/query_nl.py`, including its bridge Agent, intent
validation, operator aliases and component lanes, safe widening, `kernel_wiki` ranking, `hardware_wiki` exact lookup,
and served-record projection. Query `content` is therefore:

```json
{"query_id":"wiki-query-0123456789abcdef0123456789abcdef","records":{"stable.record.id":{"store":"gpu_wiki","wiki_id":"gpu_wiki::stable.record.id","source":"kernel_wiki","type":"technique-card","applies_to":{},"match":{},"payload":{}}},"notes":[]}
```

The complete query result passes through verbatim, including attribution IDs and all notes.
An unavailable private `internal_gpu_wiki` store is reported by upstream without preventing
queries against the public store.

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
| `GET /readyz` | Upstream tools, both indexes, and SQLite readiness. |
| `POST /v1/knowledge/query` | Strict Runtime query; `content` is upstream `query_id/records/notes`. |

## Corpus

`corpus/gpu-wiki` is ordinary content of this repository, so a checkout is immediately runnable.
Startup copies it into the ignored writable `state/gpu-wiki` store, which is what lets the corpus
tools record query feedback without modifying tracked files. Editing the corpus causes one re-copy
on the next start.

The source commit and copy boundary are recorded in [corpus/README.md](corpus/README.md).
Its original Apache-2.0 license and NOTICE are preserved beside it.

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

`max_concurrent_queries` bounds simultaneous `query_nl.py` subprocesses and defaults to `16`.
Additional requests wait for a slot. This prevents unbounded model/subprocess fan-out without
serializing unrelated read-only queries behind one global lock.

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
