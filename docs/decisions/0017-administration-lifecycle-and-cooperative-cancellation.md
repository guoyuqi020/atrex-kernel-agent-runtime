# Decision 0017: Complete the single-node administration lifecycle

English | [中文](0017-administration-lifecycle-and-cooperative-cancellation.zh.md)

## Status

Accepted and implemented. This extends Decision 0015.

## Context

The initial durable Task API could enqueue and inspect work but still required CLI access for bootstrap and Failed Epoch recovery, could not requeue an inspected failure, and could cancel only queued tasks. The Event cursor had no filters, export, retention operation, or bounded aggregate view. Killing an active Task Worker from an HTTP handler would be unsafe because a synchronous subprocess thread must retain ownership until it has reaped the complete process tree.

## Decision

The authenticated administration plane exposes idempotent Bootstrap from absolute trusted-host paths, Campaign aggregate status and quiescent cancellation, Failed Epoch recovery, failed-task requeue, correlated Event filters, bounded NDJSON export, bounded acknowledged-prefix pruning, and current Event/Task counters. All mutations use the existing Registry and Bootstrap services. The ASGI process still never launches an Agent.

Cancelling a running Task changes it to `cancelling` while retaining its Worker lease. The Task Worker observes the request during heartbeat and cancels the Scheduler scope. A synchronous Worker subprocess remains cancellation-shielded until its bounded process owner has terminated and reaped descendants; no later Attempt or Epoch work starts. Completion or failure then atomically records `cancelled`. If the Task Worker is lost, another worker finalizes the cancellation after lease expiry without relaunching the Campaign. Queued cancellation remains immediate and idempotent.

Event filters accept exact kinds and authoritative correlation IDs. Export is bounded NDJSON. Pruning deletes only a caller-acknowledged sequence prefix in a configured batch and appends a new audit event after the removed range. Registry Schema 10 intentionally rejects earlier pre-release files.

## Consequences

An upstream controller can administer the complete implemented single-node lifecycle without direct SQLite access. Running cancellation is cooperative and bounded by the current Worker operation; it is not an unsafe immediate process kill. Operators must checkpoint exported Event sequences before pruning. Decision 0018 later adds local dependency readiness; long-term telemetry storage, distributed coordination, and target-image interruption/crash acceptance remain separate requirements.
