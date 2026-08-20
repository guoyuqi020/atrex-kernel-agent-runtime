# Decision 0013: Persist branch-local Attempt Evidence

English | [中文](0013-branch-local-attempt-evidence.zh.md)

## Status

Accepted and implemented.

## Context

Every Optimizer execution is a fresh process and Session. Passing only the Epoch-start Evidence and current Branch-best Kernel prevents process-context contamination, but a later Attempt must still learn from earlier same-Trajectory evaluations, diffs, and annotations. Rebuilding that history only at launch without persisting its identity would make exact model-visible inputs ambiguous after restart or configuration changes.

## Decision

Before inserting an Attempt, the trusted Runtime seals an `ATTEMPT_EVIDENCE` artifact from the
contiguous completed prefix of lower ordinals in the same Epoch and Trajectory. It contains
authoritative input/output Kernel and Gateway facts, bounded Kernel diffs, bounded Session
projections whose `raw_files` preserve exact captured content, and explicitly untrusted final
annotations. Configured redaction applies only to normalized event summaries. It never contains a
competing Trajectory.

The Attempt row persists the Artifact Digest. Runtime projects the cumulative Lineage checkpoint and
this private snapshot into the Optimizer's unified read-only `input/evidence` view; the source
Artifacts retain independent identities for recovery and provenance. Infrastructure retries reuse
both source Digests while still receiving a fresh process, Session, and Workspace.

## Consequences

Within-Epoch learning is explicit, immutable, deterministic, Trajectory-scoped, and reconstructable without reusing model context. Attempt Evidence increases Artifact storage and remains subject to retention and garbage collection. Target Backend acceptance is still required, and cross-Branch sharing occurs only after the Epoch is committed into cumulative Lineage Evidence.
