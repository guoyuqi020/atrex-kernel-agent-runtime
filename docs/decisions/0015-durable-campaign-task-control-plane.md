# Decision 0015: Run Campaign requests through durable tasks

English | [中文](0015-durable-campaign-task-control-plane.zh.md)

## Status

Accepted and implemented; the administration lifecycle and cooperative cancellation are extended by [Decision 0017](0017-administration-lifecycle-and-cooperative-cancellation.md).

## Context

A Campaign may run for hours and launch multiple isolated Agent processes. Executing that work inside an HTTP request would couple scheduler ownership to an ASGI connection, make process loss ambiguous, and provide no durable idempotency or takeover point. Operators and upstream systems also need an authenticated way to submit work and consume ordered control-plane events without receiving direct Registry access.

## Decision

The optional administration plane exposes a versioned bearer-authenticated API. `POST /v1/admin/tasks` records an absolute Campaign target in the Registry and returns immediately. Its caller-supplied creation key is idempotent only when Campaign ID, target Epoch, and finalization flag are identical. Status lookup, queued-task cancellation, and sequence-cursor Event reads use the same API. The bearer value remains environment-owned, must contain at least 32 bytes, and is compared in constant time.

An independent `run-task-worker` process claims the oldest eligible task with a renewable lease, invokes the existing scheduler, and records a terminal result. Expired running tasks are reclaimable, so task execution is at least once. Safety does not depend on task ownership alone: scheduler writes continue to require lineage leases and fences, and Gateway operations continue to require Attempt capabilities and recovery generations. Cancellation only transitions queued tasks; it never claims to interrupt a running Agent process tree.

Registry Schema 9 intentionally rejects earlier pre-release files.

## Consequences

HTTP request lifetime is separated from Campaign execution, duplicate submissions have a stable result, a lost Task Worker can be replaced, and upstream observers can checkpoint an ordered Event sequence. A reclaimed task may repeat scheduler calls, so handlers must retain the existing idempotent absolute-target semantics. Direct Campaign bootstrap and aggregate status APIs, running-task interruption, failed-task requeue, filtered Event export, payload-version evolution, and multi-node coordination remain separate work.
