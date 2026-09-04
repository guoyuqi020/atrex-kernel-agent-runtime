# Decision 0039: Three-form Challenger proposals

## Decision

Each Evolver invocation emits exactly one uniform `EvolutionOutput` proposal. Every mode uses
`kernel_agent_revision_id`, `changed_paths`, and `contributing_paths`. The second contains sorted paths relative to the selected visible Bundle, including all six adaptive
directories. `reuse` requires an empty array; a new revision requires a real change:

- `evolved` creates a new Agent revision whose parent is the Epoch Active revision;
- `reuse` enters one visible historical revision unchanged and creates no revision or version label;
- `evolve_from_history` creates a new Agent revision whose parent is one visible historical revision.

The reported revision selects the visible Bundle and single parent; exact State identity remains
Runtime control data. For `evolve_from_history`, copy the complete selected `input/agents/agent-vN/`
into `candidate/` before editing. Runtime validates same-DSL eligibility and the complete Bundle diff.
Adaptive-directory modifications are part of `changed_paths`. Reuse requires Candidate unchanged.

Every proposal may carry bounded `unimplemented_capabilities`. Each entry names an Agent capability,
its expected Kernel-optimization benefit, and why the Evolver could not implement it. Runtime stores
these entries as untrusted Evolution Evidence for later Evolvers; they do not grant capabilities or
affect selection.

`contributing_paths` records sorted, unique workspace-relative files or directories actually incorporated from
`input/agents/agent-vN/` or `input/evidence/agent-vN/resources/`, including Parent resources from other
Trajectories. Mere reading and automatic Parent inheritance are not contributions. Paths must exist,
contain no links/traversal, and belong to eligible evaluated history or Parent, never a same-Epoch
unevaluated Challenger. `reuse` requires `[]`. Runtime records ownership and exact content snapshots
in the Evolution Trace; the field does not change the Bundle base or revision ancestry.

Proposal type, Source reference revision, resulting competing revision, and Evolution Trace Digest belong to the
Epoch Challenger participation record. Revision parentage remains a single-parent tree. Epoch
competition, reuse, and promotion are a separate timeline and do not add ancestry edges.

## Consequences

Historical designs can be retried without artificial copies or version inflation, and a promising
older design can become the base of a new revision without pretending that it descended from the
current Active. The Evolver may redesign any Optimizer-owned Candidate content as long as the final
Bundle contract remains valid. Evidence and administration views expose both ancestry and
participation provenance.
