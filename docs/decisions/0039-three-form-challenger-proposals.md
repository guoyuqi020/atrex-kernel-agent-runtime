# Decision 0039: Three-form Challenger proposals

## Decision

Each Evolver invocation emits exactly one uniform `EvolutionOutput` proposal. Every mode uses
`kernel_agent_revision_id` and `changed_paths`. The latter contains only sorted paths relative to
the selected Agent Source root. `reuse` requires an empty array; a new-revision mode may also use an
empty array for a Runtime-State-only change:

- `evolved` creates a new Agent revision whose parent is the Epoch Active revision;
- `reuse` enters one visible historical revision unchanged and creates no revision or version label;
- `evolve_from_history` creates a new Agent revision whose parent is one visible historical revision.

The reported revision identifies Agent Source only. Runtime State identity and checkpointing remain
private Runtime control data. Runtime validates every referenced Source revision against the
invocation's frozen visible set and same-DSL
Lineage. For new revisions it independently compares Candidate `source/` and `runtime-state/` with
the selected Source base and initial Active State checkpoint, validates the reported Source-only
changed-file set, and requires at least one actual Source or State change. Runtime State paths are
not part of the Agent report. For reuse it requires Source and the common seed to remain unchanged. For
`evolve_from_history`, Evolver replaces Source with a writable copy from the selected
`input/historical/agent-vN/` and may synthesize the common seed from visible historical state.
Runtime validates the declared base and checks the final Diff across Source and state seed.

Every proposal may carry bounded `unimplemented_capabilities`. Each entry names an Agent capability,
its expected Kernel-optimization benefit, and why the Evolver could not implement it. Runtime stores
these entries as untrusted Evolution Evidence for later Evolvers; they do not grant capabilities or
affect selection.

Proposal type, Source reference revision, resulting competing revision, and Evolution Trace Digest belong to the
Epoch Challenger participation record. Revision parentage remains a single-parent tree. Epoch
competition, reuse, and promotion are a separate timeline and do not add ancestry edges.

## Consequences

Historical designs can be retried without artificial copies or version inflation, and a promising
older design can become the base of a new revision without pretending that it descended from the
current Active. The Evolver may redesign any Optimizer-owned Candidate content as long as the final
Bundle contract remains valid. Evidence and administration views expose both ancestry and
participation provenance.
