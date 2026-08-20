# Decision 0016: Persist correlated lifecycle events in the Registry

English | [中文](0016-durable-correlated-lifecycle-events.zh.md)

## Status

Accepted and implemented.

## Context

Session logs and sealed traces explain Agent behavior but do not provide a small ordered control-plane history for incident response. Existing Registry transition events did not cover Worker processes, Gateway operations, GPU Wiki selection, or explicit rollback, and their payloads had no version or consistent aggregate correlation. Process-local logging alone would also lose the ordering relationship with durable lifecycle changes.

## Decision

Trusted Runtime components append events through the Registry. Every payload has `schema_version: 1` and a `correlation` object derived from authoritative relationships, containing the applicable Campaign, lineage, Epoch, Attempt, Task, Kernel Revision, or Wiki feedback identities. Payloads include bounded metadata, status, token counts, and Artifact Digests, but never secrets, prompts, model responses, or raw Session content.

Optimizer and Evolver runners record Worker start, exit, infrastructure failure or timeout, and cleanup after the owned process has been reaped. Evolver validation records sealed or rejected candidates. The Gateway Proxy records authorized submission, terminal result, and failures. The Wiki Proxy records live query submission, completion after interaction freezing, and failure. Registry selection transactions record Kernel and Kernel Agent promotion or rollback alongside the existing lifecycle events.

Events are synchronous durable facts: a required event write failure fails the surrounding trusted operation instead of silently producing an unobservable state change. The authenticated sequence-cursor API remains the read interface.

## Consequences

Operators can correlate the major single-node lifecycle without reading Worker files or duplicating model-visible content. Replayed external calls may produce multiple invocation events even when the underlying operation is idempotent; their operation and idempotency identities distinguish them. Retention, server-side filters, export formats, direct metrics aggregation, and target-image crash testing remain separate requirements.
