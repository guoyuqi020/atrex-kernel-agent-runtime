# Decision 0018: Local readiness and offline Artifact retention

English | [中文](0018-readiness-and-offline-artifact-retention.zh.md)

## Status

Accepted and implemented.

## Context

Process liveness did not distinguish a responsive ASGI loop from unavailable local durable storage. Immutable CAS objects also accumulated indefinitely, including abandoned objects sealed before a later database write failed. Deleting by directory scan alone was unsafe because authoritative references exist in both the Registry and Gateway control database, retained Events may preserve diagnostic Artifact references, and a live process can be between sealing an Artifact and committing its reference.

## Decision

`GET /readyz` runs bounded read/write probes against the Registry, Gateway control database, Agate job store, and Artifact Store staging area. It returns only failed local dependency names and never includes exception text. `/healthz` remains liveness-only. External Agate, GPU Wiki, Agent providers, cgroups, and GPUs are deliberately excluded: their temporary unavailability must not prevent the trusted control plane from starting for inspection and recovery.

Artifact retention is an explicit offline CLI operation. The collector unions every Artifact Digest referenced by Registry columns, committed Gateway outcomes, and retained Runtime Event payloads, then follows existing Digest tokens embedded in those verified Artifacts to a transitive CAS closure. It considers only unreachable objects older than an operator-supplied minimum age, verifies each complete CAS object before deletion, fails on unexpected entries, and stops at an operator-supplied object limit. Dry-run is the default. Applying deletion requires `--confirm-runtime-stopped`; all Runtime, Worker, Bootstrap, and Wiki drainer processes must actually be stopped because the confirmation is an operational precondition, not an inferred lock.

An applying pass emits `artifact.gc_completed` after deletion. The operation is intentionally not transactional across filesystem deletion and SQLite audit append; the deleted objects were absent from the complete durable-reference snapshot, and a missing audit Event does not make them referenced.

## Consequences

Load balancers can remove a process whose owned durable stores are unavailable without coupling startup to external services. Disk retention is bounded by an auditable, conservative maintenance operation rather than manual CAS deletion. Operators must preserve a consistent backup, keep the capability signing key available to open Gateway state, use a deployment-specific age, and prove quiescence before every applying pass. Target-image readiness failure and GC/restore rehearsals remain production acceptance requirements.
