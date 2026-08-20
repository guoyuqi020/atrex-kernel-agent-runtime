# ADR 0045: Provider-native usage accounting

## Status

Accepted.

## Decision

Runtime accounts for each Worker in the unit reported authoritatively by its selected Provider.
QoderCLI uses `credits`; Claude, Codex, and Pi use disjoint provider-token buckets. Runtime never
estimates tokens from Qoder credits and never treats Qoder's zero-filled token fields as usage.

Core receives `ATREX_USAGE_UNIT` and `ATREX_USAGE_BUDGET`. Its schema-v2 report records `budget`,
`consumed`, zero-or-real token buckets, and optional `credits`. Qoder credit deltas are deduplicated
by provider message ID for live quota enforcement, while `result.total_credits` (or the equivalent
model-usage total) is the authoritative terminal value. The Evolver remains unbounded but emits the
same report with `budget=null`.

Worker sessions, Bootstrap generations, Attempt traces, Runtime events, and the unredacted Session
trace retain the selected unit and consumption. Configuration exposes both `max_session_tokens` and
`max_session_credits`; only the field matching the selected Backend is enforced.

## Consequences

- Qoder can run under a fail-closed quota without fabricated token counts.
- Cross-provider totals must always be grouped by unit; credits and tokens are not additive.
- A Qoder message that declares usage but omits credits is incomplete accounting and fails closed.
