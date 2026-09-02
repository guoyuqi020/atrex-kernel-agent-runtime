# ADR 0036: Filesystem-only Evolver evidence

English | [中文](0036-frozen-evolver-runtime-tools.zh.md)

## Status

Accepted and implemented. This supersedes the earlier Evolver Runtime Tools design.

## Context

Evolver performs offline Agent engineering over one immutable Lineage checkpoint. Unlike Optimizer,
it does not evaluate Kernels, query Wiki, append journals, or otherwise interact with mutable Runtime
state. A dedicated command client, HTTP capability, query Catalog, and Candidate reset record duplicate
facts that Runtime can materialize as ordinary read-only files.

## Decision

Evolution Input schema v11 has no Runtime query authority. Runtime materializes:

- every visible Agent repository and per-Trajectory Runtime State once under
  `input/agents/agent-vN/`, keyed by stable Lineage version rather than temporary Epoch role;
- one Runtime-derived optimization summary per visible version under `input/evidence/agent-vN/`,
  plus one Conversation and Attempt Report per Attempt for every Branch in the latest completed
  Epoch;
- available prior Agent-creation reports under `input/evolution-reports/evo-N.json`; and
- a read-only catalog that identifies the current Parent and any same-Epoch Challengers already
  created but not yet evaluated.

Evolver reads these files directly and may modify only Candidate `source/`, Candidate
`runtime-state/`, and `scratch/`. Runtime initially copies Active Source and the latest completed
Epoch winner's best-Kernel Trajectory terminal State after that Epoch's last Attempt. The next
Epoch's Active Branch uses the same State seed. Missing terminal State falls back to that
Trajectory's Epoch-start State, the revision seed, and empty default. For `evolve_from_history`,
Evolver replaces Source with the selected historical Source and may synthesize the common seed from
visible historical state. A new revision may also combine Source, Skills, or Tools from multiple
eligible visible Agents and records those contributors separately from its single Source base. Runtime verifies eligibility
and validates the reported Source-root-relative Diff plus its private State Diff. Every new Revision seals Source and State as one logical
Bundle. No Candidate Base side
record is trusted or required.

The Evolver query endpoint, query capability, public helper, Candidate allowlist, and private query
snapshot are removed. Detailed history and full Evolution traces remain in Runtime's existing
Evidence and Registry stores; only compact Agent-authored creation reports are projected.

## Consequences

- Evolver's entire input contract is inspectable as a frozen filesystem tree.
- Workspace and Prompt complexity are reduced.
- No Evolver-scoped HTTP credential or query service exists.
- Historical-base replacement is Agent-authored but independently validated by Runtime.
- Multi-Agent contribution is explicit provenance and does not add ancestry edges.
- Optimizer Runtime Tools remain unchanged because Optimizer still needs live services and journals.
