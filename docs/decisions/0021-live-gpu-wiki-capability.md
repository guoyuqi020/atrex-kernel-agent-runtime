# Decision 0021: Query GPU Wiki live, freezing interactions before return

English | [中文](0021-live-gpu-wiki-capability.zh.md)

## Status

Accepted and implemented. This is the current Wiki integration decision.

## Context

GPU Wiki is an external knowledge source, while lineage experience is Agent-produced local history. Treating one preselected Wiki snapshot as the next Epoch's common memory conflated those responsibilities and prevented an Optimizer from asking a focused question when it became relevant. Returning a live answer without first making it durable would make the Session Trace irreproducible.

## Decision

Each configured Optimizer receives a `wiki-query` Core tool plus an Attempt-scoped
Runtime capability. Query sends only its immutable manifest Attempt ID, a focused question, and an
idempotency key to `POST /v1/wiki/query`; its Agent-facing content is GPU Wiki's exact
`records`/`notes` projection. A stable `records` mapping key is the Record ID and each value is the
complete safe served Record. The trusted Runtime reconstructs Campaign, lineage,
Epoch, branch, ordinal, Kernel Agent revision, operator, DSL, hardware, Evaluation Contract, Epoch
Evidence, and Attempt Evidence context from authoritative stores. Only Runtime holds the external
Wiki credential.

Runtime validates strict Wiki responses and canonical content digests, seals every complete trusted Query interaction as a `WIKI_INTERACTION` Artifact, and commits the Artifact Digest to the idempotency reservation before returning content to the Worker. The Core tool projects only knowledge `content`; protocol versions, interaction/snapshot identities, and integrity Digests remain internal. An identical same-key retry replays the frozen response without another external operation; a changed request fails. Wiki Query does not consume the Gateway benchmark-call quota, but remains bounded by request and response bytes, transport and process timeouts, and the provider-token-only Agent budget.

Runtime does not inject Wiki selections into cumulative Evidence and does not store lineage experience in Wiki. Wiki integration is query-only: Runtime never uploads consumption records, Session traces, Kernel history, winner facts, component evolution, or lineage memory.

## Consequences

An Optimizer receives the upstream Wiki's complete safe Record projections from Query and preserves
the stable IDs of Records that materially inform its work. Every model-visible answer remains
attributable and replayable. External service credentials, trusted context, and audit identities remain outside the
model context. Real Wiki availability can still affect an individual tool call, so
Agents and prompts must handle explicit failures rather than receiving a silent cached substitute.
