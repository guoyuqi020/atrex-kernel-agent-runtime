# Atrex Local GPU Wiki

English | [中文](README.zh.md)

This workspace is a local HTTP adapter for the independent Atrex GPU Wiki. It is for Runtime
integration tests only. Query behavior is executed by the commit-pinned upstream implementation.

The adapter does not implement its own retrieval algorithm. For every query it executes the
commit-pinned upstream `gpu-wiki/tools/query_nl.py`, including the upstream bridge Agent, intent
validation, normalization, safe widening, `kernel_wiki` ranking, `hardware_wiki` exact lookup,
public/internal Store isolation, and served-record projection. Query `content` is therefore exactly:

```json
{"records":{"stable.record.id":{"store":"gpu_wiki","source":"kernel_wiki","type":"technique-card","applies_to":{},"match":{},"payload":{}}},"notes":[]}
```

Runtime continues to provide the versioned HTTP envelope, digest verification, Attempt authority,
and freezing. Every `records` mapping value is already the complete safe
served Record. Consumers preserve its stable mapping key when the Record materially informs work;
there is no separate read operation.

The vendored pinned source remains immutable. Startup atomically synchronizes it into
`state/gpu-wiki`. Runtime treats the Wiki as a read-only external knowledge source.

## HTTP interface

| Method and path | Result |
| --- | --- |
| `GET /` or `GET /ui` | Local browser query client. |
| `GET /healthz` | Process liveness. |
| `GET /readyz` | Upstream tools, both indexes, and SQLite readiness. |
| `POST /v1/knowledge/query` | Strict Runtime query; `content` is upstream `records/notes`. |

## Pinned source

`reference.lock.json` pins Alibaba `atrex-kernel-agent` commit
`71b16928579474c93039053d2facfeaf7134e268`. Its exact `gpu-wiki` tree is vendored at
`corpus/gpu-wiki`, so a Runtime checkout is immediately runnable without a second Git clone.
Maintainers refresh the vendored tree from the locked commit; Runtime execution never downloads or
modifies it.

```bash
git -C /path/to/atrex-kernel-agent archive \
  71b16928579474c93039053d2facfeaf7134e268 gpu-wiki
```

## Run

The checked-in config does not override upstream query defaults. Therefore the pinned
`query_nl.py` selects its own default bridge CLI, timeout, and Record cap. Optional `agent_cli`,
`query_timeout_seconds`, and `max_results` fields are explicit HTTP deployment overrides; no model
credential is stored by local-wiki.

```bash
PYTHONPATH=workspaces/local-wiki/src \
  .venv/bin/python -m atrex_local_wiki serve \
  --config workspaces/local-wiki/configs/local.example.json
```

Open [http://127.0.0.1:8091/](http://127.0.0.1:8091/). When overriding `agent_cli`, use a backend
accepted by the pinned upstream `tools/agent_launch.py`.

## Verify

```bash
PYTHONPATH=src:workspaces/local-wiki/src .venv/bin/pytest workspaces/local-wiki/tests
PYTHONPATH=workspaces/local-wiki/src \
  .venv/bin/ruff check workspaces/local-wiki/src workspaces/local-wiki/tests
PYTHONPATH=workspaces/local-wiki/src \
  .venv/bin/mypy --config-file workspaces/local-wiki/pyproject.toml \
  workspaces/local-wiki/src
```
