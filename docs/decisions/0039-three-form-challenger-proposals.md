# Decision 0039: Three-form Challenger proposals

## Decision

Each Evolver invocation emits exactly one tagged `EvolutionOutputV3` proposal:

- `evolved` creates a new Agent revision whose parent is the Epoch Active revision;
- `reuse` enters one visible historical revision unchanged and creates no revision or version label;
- `evolve_from_history` creates a new Agent revision whose parent is one visible historical revision.

Runtime validates every referenced revision against the invocation's frozen visible set and same-DSL
Lineage. For new revisions it independently compares the complete Candidate repository with the
selected base and requires the exact declared changed-file set. For reuse it requires the seeded
Candidate to remain unchanged. `evolve_from_history` additionally requires the constrained Runtime
`candidate-reset` operation and a matching Candidate-base record.

Every proposal may carry bounded `unimplemented_capabilities`. Each entry names an Agent capability,
its expected Kernel-optimization benefit, and why the Evolver could not implement it. Runtime stores
these entries as untrusted Evolution Evidence for later Evolvers; they do not grant capabilities or
affect selection.

Proposal type, base revision, resulting competing revision, and Evolution Trace Digest belong to the
Epoch Challenger participation record. Revision parentage remains a single-parent tree. Epoch
competition, reuse, and promotion are a separate timeline and do not add ancestry edges.

## Consequences

Historical designs can be retried without artificial copies or version inflation, and a promising
older design can become the base of a new revision without pretending that it descended from the
current Active. The Evolver may redesign any Optimizer-owned Candidate content as long as the final
Bundle contract remains valid. Evidence and administration views expose both ancestry and
participation provenance.
