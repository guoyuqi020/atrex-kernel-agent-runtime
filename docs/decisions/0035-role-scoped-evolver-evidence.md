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

Each Evolver Epoch preserves Active, Challenger, winner Agent, starting Kernel, and best Kernel
identities. `branches/` contains every Attempt summary, report, diff, and an unredacted derived copy
of each retained Session Artifact. Authoritative Session retention omits only high-frequency Claude
`system/thinking_tokens` estimate telemetry; the derived copy applies the same rule to older
Artifacts. `evolution/` contains every available Challenger Evolver Session. `kernels/index.json`
records roles and authoritative outcomes, while each referenced exact Kernel Artifact is
materialized once under `kernels/<kernel-revision-id>/`.

The cumulative source Evidence stores structured starting and best Kernel facts in addition to their
IDs. All Attempt output Kernels carrying Artifact Digests are projected.

## Consequences

- Evolver can compare successful and failed Agent designs against exact Kernel/evaluation history.
- Optimizer branch isolation and fresh-session semantics do not change.
- Evolver Evidence is larger and may repeat a retained Kernel across adjacent Epoch views; exact
  content-addressed source identity remains explicit.
- The Evolver Bundle validates `all_completed_branches` before using the projected history.
