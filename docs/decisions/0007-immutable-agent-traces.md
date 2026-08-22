# Decision 0007: Persist Agent runs as immutable Trace Artifacts

English | [中文](0007-immutable-agent-traces.zh.md)

## Status

Accepted and implemented.

## Context

Every Optimizer and Evolver invocation uses a fresh process and Session. Runtime needs durable
provenance for audit, retry attribution, and Evidence without carrying
hidden conversational continuity into later Sessions. Raw provider history is too large for SQLite,
and a single Trace field would overwrite retries.

## Decision

Artifact Store owns original Session Trace trees and structured Evolution records. Registry stores
their content digests plus Worker lifecycle, role, model, workspace, terminal reason, configured
Optimizer token budget, and complete provider-reported token buckets. A stable Attempt or Evolution
subject may own multiple append-only Worker Sessions.

Runtime creates the Worker Session record before launch, updates it through terminal state, and
seals any available raw Trace without redaction. Evidence stores normalized summaries and source
digests; Agent views materialize the original Trace Artifact explicitly. Trace content is audit input, not an
authoritative success claim or automatic Agent memory.

## Consequences

Restart and retry preserve attribution without reusing model context. A process that fails before
any Trace can be captured still has a terminal Worker Session record with failure identity, but no
fabricated Trace Artifact. Runtime-authoritative Gateway records and Registry transitions, not
provider output, determine Kernel and Agent promotion.
