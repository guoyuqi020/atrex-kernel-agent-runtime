# Runtime code organization

[中文](code-organization.zh.md) | English

This document defines the maintainability boundary of the Runtime source tree. It is an implementation guide, not a protocol: moving code between these modules must not change durable identities, SQLite schemas, Artifact formats, Worker layouts, or HTTP/CLI responses.

## Organizing principle

Dependencies should point inward:

```text
entrypoints (CLI / ASGI)
        |
        v
composition + presentation
        |
        v
application services (bootstrap / controller / workers)
        |
        v
domain + ports
        ^
        |
infrastructure adapters (registry / artifacts / Agate / Wiki / Git)
```

- Entrypoints parse input, call one application operation, and render a result. They do not assemble SDK credentials or duplicate domain projections.
- Composition modules own configuration-to-object wiring. They contain no business decisions.
- Presentation functions convert durable models into stable JSON read models shared by CLI and HTTP.
- Application services implement use cases and depend on ports where replacement or testing is useful.
- Infrastructure adapters may depend on domain types; domain code must not import SQLite, HTTP, subprocess, or SDK implementations.

## Current source layout

The existing top-level directories already represent useful stable boundaries and should be retained:

| Area | Responsibility |
|---|---|
| `api/` | authenticated administration HTTP and ASGI lifecycle |
| `cli/` | public command entrypoint, parser, command families, progress rendering |
| `composition/` | configuration-to-object assembly for Bootstrap, Campaign, and Wiki workers |
| `domain/` | immutable identities, state models, and domain errors |
| `controller/` | Campaign/Epoch orchestration, leases, evidence assembly, durable tasks |
| `workers/` | Optimizer/Evolver process, workspace, sandbox, and report handling |
| `gateway/` | Agate adapter, capability control, proxy, evaluation, and metrics |
| `registry/` | durable state port and SQLite implementation |
| `artifacts/` | content-addressed Artifact storage |
| `knowledge/` | external Wiki query protocol and proxy |
| `kernel_agents/` | Git import and immutable Agent revision construction |

Shared leaf seams keep repeated mechanics out of application services:

- `presentation.py` owns JSON projections used by both administration HTTP and CLI inspection.
- `gateway/configuration.py` is the single Agate settings/secret-to-connection resolver.
- `asgi.py`, `filesystem.py`, and `serialization.py` own bounded transport, safe-file, and canonical
  JSON primitives.
- `gateway/candidate.py` owns verified Kernel Artifact resolution.

`composition/bootstrap.py` is the canonical composition point for Campaign bootstrap, base Agent loading, and Artifact-rooted Lineage seeding. CLI and ASGI no longer construct those graphs independently.

The second cleanup pass separated command and control responsibilities without changing the public CLI or HTTP contracts:

- `cli/parser.py` owns the complete argument schema. The remaining `cli/` modules own their command families, while `cli/__init__.py` retains only serving and dispatch. The installed `atrex_runtime.cli:main` entrypoint is unchanged, and `cli/__main__.py` also supports module execution.
- `controller/tasks.py` owns durable Task leasing/heartbeat execution; `api/administration.py` remains the authenticated HTTP control plane and `api/app.py` owns ASGI lifecycle composition.
- `composition/bootstrap.py`, `composition/campaign.py`, and `composition/knowledge.py` contain deployment-object assembly without adding compatibility shims at the old flat paths.
- `gateway/control_models.py` owns immutable capability, Bootstrap, and evaluation records; `gateway/control_schema.py` owns schema creation and legacy migrations; `gateway/control.py` retains the SQLite-backed authority operations and compatible exports.
- `gateway/execution.py` centralizes blocking SDK execution, JSON-object validation, structured candidate rejection, and Eval/Profile result sealing for Finalization, Lineage Seed, and Measurement.

The final reduction pass removed private CLI forwarding aliases, consolidated repeated terminal-table rendering, and moved the shared Agate evaluation-request construction into `gateway/execution.py`. These are code-size reductions rather than new abstraction layers; persisted records and external interfaces are unchanged.

Evaluation policy remains in Finalization, Lineage Seed, and Measurement even though their common
Agate execution and result sealing live in `gateway/execution.py`. Cross-cutting leaf modules such
as `config.py`, `ports.py`, and `secrets.py` remain at the package root; a generic utility directory
would obscure their dependency role.

## What should not be merged

- Attempt evidence and Epoch/Evolver evidence have different visibility and trust semantics.
- Optimizer and Evolver workspace/process code have different authority and must remain separate.
- Gateway exploratory evaluation and Runtime authoritative finalization have different state effects even if they later share an execution helper.
- Registry and Artifact Store have different consistency models and should not become one persistence abstraction.

## Refactoring gates

Every structural change must pass Ruff, strict mypy, and the complete test suite. Database migrations, protocol fields, configuration keys, CLI output, or Worker-visible layouts require a separate design change rather than being hidden inside code cleanup.
