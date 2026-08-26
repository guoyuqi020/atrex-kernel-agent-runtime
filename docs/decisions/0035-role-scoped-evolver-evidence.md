# ADR 0035: Role-scoped completed-Epoch Evidence

English | [中文](0035-role-scoped-evolver-evidence.zh.md)

## Status

Accepted and implemented.

## Context

The Optimizer needs branch isolation so a fresh Attempt cannot copy competing search paths. The
Evolver has a different job: improve the Agent design after observing whether Active and Challenger
designs actually produced better Kernels. Showing it only the promoted branch hides negative
results, losing designs, evaluated Kernel sources, and the evidence needed to distinguish a useful
Agent change from a lucky Kernel.

## Decision

`EvidenceViewManifestV1.visibility.completed_epochs` is role-scoped:

- Optimizer uses `promoted_lineage`; completed Epochs remain stripped of branch-control identity,
  and the current Epoch exposes only earlier Attempts from the same Trajectory.
- Evolver uses `all_completed_branches`; it has no in-progress Epoch and sees every completed Active
  and Challenger branch.

The durable Evidence store preserves Active, Challenger, winner Agent, Attempt outcomes, Session
Artifacts, and exact Kernel identities for every completed branch. Evolution workspace projection is
deliberately smaller: for each currently competing Agent, Runtime materializes an authoritative
optimization summary plus one unredacted Conversation per Attempt from that Agent's latest completed
Epoch. Completed non-current Agent versions receive the same compact summary alongside their source.
Older detailed Epoch trees remain available only in Runtime's Registry and Artifact stores.

## Consequences

- Evolver can compare successful and failed Agent designs using compact authoritative outcomes and
  the latest relevant Conversations.
- Optimizer branch isolation and fresh-session semantics do not change.
- Runtime retains complete audit history without duplicating it into every Evolution workspace.
- The frozen filesystem input remains branch-complete for current competition while bounded in size.
