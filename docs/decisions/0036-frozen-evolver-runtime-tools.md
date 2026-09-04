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

- one complete Bundle per visible Agent under
  `input/agents/agent-vN/`, keyed by stable Lineage version rather than temporary Epoch role;
- one Runtime-derived optimization summary and `resources/trajectories/` snapshots per version under `input/evidence/agent-vN/`,
  plus one Conversation and Attempt Report per Attempt for every Branch in the latest completed
  Epoch;
- available prior Agent-creation reports under `input/evolution-reports/evo-N.json`; and
- a read-only catalog that identifies the current Parent and any same-Epoch Challengers already
  created but not yet evaluated.

Evolver reads these files directly and may modify only the complete `candidate/` Bundle and report
scratch space. The Parent Bundle combines the winning implementation and its best-Kernel
Trajectory's terminal resources; the next Active starts from the same snapshot. Missing terminal
State falls back to Epoch-start State, revision seed, and packaged defaults.
Historical derivation copies the complete selected Bundle before editing it. Evolver may synthesize
eligible Agents' resources and record their contributions. Runtime validates the selected revision
and entire Bundle diff, including all six adaptive directories, then seals Bundle and State.

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
