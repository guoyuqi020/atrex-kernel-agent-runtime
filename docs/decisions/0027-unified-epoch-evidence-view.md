# ADR 0027: Unified Epoch-organized Evidence view

English | [中文](0027-unified-epoch-evidence-view.zh.md)

## Status

Accepted and implemented.

## Context

Runtime persists a cumulative Epoch-start `EVIDENCE` checkpoint and a same-Trajectory
`ATTEMPT_EVIDENCE` snapshot with different lifecycles and Digests. Exposing those storage objects as
separate trees would leak control-plane mechanics to Agents and force Prompts to reconcile
overlapping histories.

## Decision

Runtime keeps the source Artifacts independent but projects their authorized data into one
read-only tree rooted at `input/evidence`. The internal `.runtime/evidence-manifest.json` binds the
Agent role, Lineage checkpoint, last completed Epoch, optional in-progress snapshot, and visibility
policy; `.runtime/evidence-instructions.md` is validated and injected into the Prompt. Neither is
part of the Agent-facing Evidence tree. `bootstrap/` and `epochs/NNNNNNNN/` form one chronology.

For an Optimizer, a completed Epoch contains only the promoted Agent revision's Attempts, while the
current Epoch contains only lower-ordinal Attempts from the selected Trajectory. For an Evolver,
completed Epochs contain all completed Active and Challenger branches as specified by ADR 0035.

Every Optimizer Epoch uses the same `trajectories/<ordinal>/attempts/<ordinal>/` hierarchy; it has no
Agent-visible `branches/` directory or Active/Challenger task field. Attempts are serial within one
Trajectory, advancing from the latest retained Kernel, while sibling Trajectories are independent
parallel searches from the same Epoch-start Kernel. Epochs themselves are serial: Bootstrap seeds
Epoch 1, and a completed Epoch independently promotes the next active Agent revision and the best
retained correct Kernel. Consequently the next Epoch's Agent and starting Kernel need not have the
same producer; when no Candidate improves the Kernel, the previous starting Kernel carries forward.
Each visible Attempt directory contains only
the Runtime Final `report.json` and the latest sealed `conversation.jsonl`. Runtime storage retains
all Session, Kernel, Trial, Gateway Result, summary, and diff Artifacts; exact data is recovered
through Runtime tools rather than duplicated into the Optimizer filesystem. Agent and Evolver
annotations remain explicitly untrusted. Snapshot selection is Manifest metadata, never a separate
Evidence root or a replacement for a source Digest. Evolver Evidence remains a richer all-branch
diagnostic view. Cross-branch Direction and Experiment Journals used by list/load tools are appended
immediately to Runtime-owned Attempt tables; terminal Report Artifacts freeze the handoff snapshot
and remain a compatibility fallback. They are resolved on demand through Attempt-scoped Runtime
queries; no Journal history projection is written into the Workspace. Evolver traces are never
projected into Optimizer Evidence.

Optimizer Epoch directories contain only `trajectories/`: Runtime does not project Epoch
`summary.json`, `lessons.json`, or `measurements.json`. Scheduling state remains in the Registry,
the current Kernel is authoritative under `input/kernel/`, and exact measurements remain bound to
Gateway Result Artifacts. Evolver Evidence retains its richer aggregate files for diagnosis.

Runtime writes one `instructions.md` Prompt Fragment beside the View Manifest. The Fragment describes
the exact structure and reading rules without assigning the layout a separate version name. The
Manifest binds its SHA-256 and each process receives its fixed path. Core and Evolver repositories
contain generic verification and concatenation hooks, so structure text has one Runtime-owned
source.

Optimizer and Evolution Input Manifests both bind `input/evidence`; Runtime retains the independent
source Digests for recovery and provenance and validates the projected View for the receiving role.

## Consequences

- Optimizer and Evolver use one Evidence contract and one chronological root.
- Registry recovery, idempotency, Artifact retention, and Trajectory isolation retain independent
  source identities.
- Active/Challenger remains a trusted scheduling and promotion concept but is absent from the
  Optimizer filesystem, Manifest task context, and Prompt.
- Agent-visible traces may contain prompts, reasoning, tool inputs and outputs, command output,
  credentials, and other sensitive content captured by the Backend.
- The View is a deterministic projection and may be rebuilt from persisted source Artifacts.
