# Decision 0022: Keep the local GPU Wiki as a wire-compatible test workspace

English | [中文](0022-local-wiki-test-double.zh.md)

## Status

Accepted and implemented.

## Context

Runtime now owns strict live Query and post-Epoch Feedback clients, but the production GPU Wiki is developed and deployed independently. Unit fakes prove local control flow, not that a replaceable HTTP service accepts exact trusted requests, returns digest-valid responses, and enforces feedback idempotency. Production Runtime must remain independent of the reference `atrex-kernel-agent` source tree; the local test service may deliberately pin it as test data.

## Decision

Add `workspaces/local-wiki` as an independently packaged Python test service. Its production modules do not import `atrex_runtime` or `atrex-kernel-agent`. They independently implement the version-1 external request/response fields, canonical JSON digest algorithm, HTTP statuses, optional bearer authentication, and feedback identity semantics. Cross-workspace tests use Runtime's authoritative models to detect drift in either copy.

The local adapter keeps only the HTTP envelope and executes the commit-pinned reference Wiki's own
`tools/query_nl.py`; it does not reimplement intent extraction, normalization, widening, ranking,
hardware lookup, Store isolation, or served-record projection. Query therefore returns the upstream
`records`/`notes` interface, including complete safe served Records keyed by stable ID. A workspace
fixture makes unit tests independent of the reference checkout.

The pinned reference remains immutable and is atomically synchronized into a writable local Store.
SQLite retains exact Query and Feedback HTTP observations and their idempotency state. After Epoch
completion, every public kernel Record contained in a frozen interaction becomes an upstream
`served` event through the pinned `ingest_feedback.py` implementation; the pinned
`rebuild_importance.py` then folds the append-only log into ranking. The adapter does not infer
`applied`, `effective`, or `ineffective` because the Runtime report has no authoritative per-Record
adoption field. Interrupted applications remain pending and replay with stable upstream event keys.

Runtime accesses this service only through the same `gpu_wiki.base_url` and credential configuration used for the remote Wiki. The local response content has an implementation-specific version, but it remains opaque JSON under the external v1 envelope. No local-only endpoint or import is added to Runtime.

## Consequences

Developers can exercise all external Wiki paths without a remote deployment, including Query, strict request parsing, digest verification, authentication, response status, at-least-once feedback replay, upstream served-event ingestion, and ranking rebuild. Switching to production is a configuration change. Search quality, remote persistence, additional production feedback processing, availability, and operational behavior still require real-service acceptance; this test double must never be presented or deployed as the Wiki itself.
