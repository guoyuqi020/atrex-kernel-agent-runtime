# Decision 0029: Retain every Bootstrap execution generation

English | [中文](0029-append-only-bootstrap-generations.zh.md)

## Status

Accepted and implemented.

## Context

A Campaign Bootstrap has one stable, deterministic `bootstrap_attempt_id`, but a provider failure,
token-budget termination, Runtime restart, blocked Agent report, or expired authority may require a
fresh Core Session. Capability generation fencing made retries safe, but the earlier Gateway schema
overwrote the current generation and deleted uncommitted operation reservations. Raw run directories
often survived, yet no durable record related a failed Session, token use, report, Trace, workspace,
and failure reason. Operators could not reliably inspect why each physical execution ended.

## Decision

Gateway Control schema 5 stores an append-only `bootstrap_runs` row keyed by
`(bootstrap_attempt_id, recovery_generation)`. Issuance creates an `issued` row; workspace creation
binds its `run_id` and exact path; every normal return or caught failure commits one terminal
`completed` or `failed` row. The record retains finish and failure reasons, timestamps, four
provider-token buckets and budget, Session Trace and terminal Report digests, and authoritative
candidate/result digests when available. A Runtime crash that leaves a row non-terminal is marked
`superseded-by-retry` when the next generation is issued.

Gateway operations are keyed by `(attempt_id, recovery_generation, idempotency_key)`. Rotation
invalidates the old bearer and resets current quota without deleting earlier operations. Outcome
commit remains fenced by the exact authorization generation, and an existing authoritative outcome
is recovered without creating another run. Artifact GC treats all run and operation digests as live.

Authenticated administration endpoints and the `list-bootstrap-runs` / `show-bootstrap-run` CLI
expose these records without exposing capability tokens. Failed Bootstrap exceptions also identify
the Attempt, generation, and physical run. Migration from schema 4 preserves current operations and
creates one explicitly labelled legacy record for each pre-existing Bootstrap subject; history that
the old schema had already overwritten cannot be reconstructed.

## Consequences

Capability generation remains a security fence and is now also a durable execution identity. Failed
generations are queryable and auditable without treating their Agent conclusions as authoritative.
Storage grows with executions and operations, so retention must remove workspaces and Artifacts only
through policy-aware maintenance. Target deployment still requires forced-crash tests across every
issuance, Session, Gateway, and terminal-record boundary.
