# Decision 0031: Account Evolver tokens without enforcing a token quota

English | [中文](0031-unbounded-evolver-token-accounting.zh.md)

## Status

Accepted and implemented.

## Context

Optimizer Sessions need a deployment-owned cost boundary, but Evolver is invoked once per Epoch and
must be allowed to complete its Agent-revision hypothesis without a provider-token cutoff. The old
configuration exposed the same positive `max_session_tokens` field for both roles and terminated the
Evolver process group with exit 125 when the cumulative total reached that value.

## Decision

Optimizer Core phases retain their positive per-Session token quota. Evolver configuration no longer
contains `max_session_tokens`, Runtime no longer injects `ATREX_TOKEN_BUDGET`, and the Evolver never
terminates because of provider token consumption. Wall-time, output-size, process, and workspace
safety limits remain active.

Provider accounting remains mandatory. The Evolver continues to deduplicate stream events, prefer
terminal usage, and report uncached input, output, cache-read, and cache-write buckets. Its strict
TokenUsageReportV1 uses `budget_tokens=null` and `budget_exhausted=false`; Runtime requires those
values and still rejects missing or incomplete provider usage.

## Consequences

Evolver cost is observable but unbounded by token count. Operators must use provider controls and
wall-time limits as external safety mechanisms. Existing configuration containing Evolver
`max_session_tokens` is rejected by the strict schema and must be regenerated. Optimizer phases
retain their separate positive provider-token quota.
