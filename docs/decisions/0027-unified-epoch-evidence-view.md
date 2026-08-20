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

Runtime keeps the source Artifacts independent but projects them into one read-only
`EvidenceViewManifestV1` tree rooted at `input/evidence`. `manifest.json` binds the Agent role,
Lineage checkpoint, last completed Epoch, optional in-progress snapshot, and visibility policy.
`bootstrap/` and `epochs/NNNNNNNN/` form one chronology.

For an Optimizer, a completed Epoch contains only the promoted Agent revision's Attempts, while the
current Epoch contains only lower-ordinal Attempts from the selected Trajectory. For an Evolver,
completed Epochs contain all completed Active and Challenger branches as specified by ADR 0035.

An Optimizer Epoch contains `attempts/` directly; it has no Agent-visible `branches/` directory or
Active/Challenger task field. Each Attempt directory may contain a trusted summary, bounded Kernel
diff, structured report, and complete original Session Artifact directories. Runtime resolves the
selected source Digests and materializes those trace trees without redaction, filtering, or text
rewriting. Agent and Evolver annotations remain explicitly untrusted. Snapshot selection is
Manifest metadata, never a separate Evidence root or a replacement for a source Digest.

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
