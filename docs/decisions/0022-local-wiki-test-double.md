# Decision 0022: Keep the local GPU Wiki as a wire-compatible test workspace

English | [中文](0022-local-wiki-test-double.zh.md)

## Status

Accepted and implemented.

## Context

Runtime owns a strict live Query client, but the production GPU Wiki is developed and deployed independently. Unit fakes prove local control flow, not that a replaceable HTTP service accepts exact trusted requests and returns digest-valid responses. Production Runtime must remain independent of the reference `atrex-kernel-agent` source tree; the local test service may deliberately pin it as test data.

## Decision

Add `local-wiki` as an independently packaged Python test service. Its production modules do not import `atrex_runtime` or `atrex-kernel-agent`. They independently implement the version-1 external request/response fields, canonical JSON digest algorithm, HTTP statuses, and optional bearer authentication. Cross-workspace tests use Runtime's authoritative models to detect drift in either copy.

The local adapter keeps only the HTTP envelope and executes the commit-pinned reference Wiki's own
`tools/query_nl.py`; it does not reimplement intent extraction, normalization, widening, ranking,
hardware lookup, Store isolation, or served-record projection. Query therefore returns the upstream
`query_id`/`records`/`notes` interface, including canonical `wiki_id` fields and complete safe served
Records keyed by stable ID. Contents and notes pass through unchanged; operator aliases and
component decomposition are owned by the copied upstream implementation. A workspace
fixture makes unit tests independent of the reference checkout. Integration tests execute the
vendored query tools and corpus, replacing only the model process.

The pinned reference remains immutable and is atomically synchronized into a writable local Store.
SQLite retains exact Query observations. The service has no feedback endpoint.

Runtime accesses this service only through the same `gpu_wiki.base_url` and credential configuration used for the remote Wiki. The local response content has an implementation-specific version, but it remains opaque JSON under the external v1 envelope. No local-only endpoint or import is added to Runtime.

## Consequences

Developers can exercise Query, strict request parsing, digest verification, authentication, and response status without a remote deployment. Switching to production is a configuration change. Search quality, availability, and operational behavior still require real-service acceptance; this test double must never be presented or deployed as the Wiki itself.
