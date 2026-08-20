# Decision 0030: Separate exploratory evaluations from trusted finalization

English | [中文](0030-exploratory-and-authoritative-evaluations.zh.md)

## Status

Accepted and implemented.

## Context

An Optimizer needs compilation, correctness, and performance feedback while it edits a Kernel. One
Attempt can therefore evaluate several distinct candidate trees before nominating its final one.
The previous control path committed the first `evaluate` response directly as the Attempt outcome.
That made a repair impossible: an early incorrect candidate permanently occupied the outcome slot,
and a later correct candidate conflicted with it. It also conflated evidence visible to untrusted
Agent code with the result used by trusted retention and promotion logic.

## Decision

Every Agent `evaluate` call is exploratory. Runtime seals the exact candidate directory as a Kernel
Artifact, seals the complete raw Gateway response, and appends one immutable
`GatewayEvaluationRecord` containing Attempt, recovery Generation, ordinal, source `agent`,
idempotency key, candidate/result digests, correctness, latency, external job identity, and time.
Distinct requests use distinct idempotency keys; an identical retry replays the existing response
and record without another external submission. Exploratory records never commit an Attempt
outcome.

`candidate_ready` or `baseline_ready` nominates the exact final `work/kernel` tree. Runtime seals the
tree itself and requires a correct exploratory record for those exact bytes. Bootstrap submits a
fresh Agate evaluation with Runtime-owned credentials and a stable Runtime-final idempotency key.
An optimization Attempt instead provisionally registers the exact nominated Artifact and delegates
final authority to Kernel retention. Ordinary Evaluate measures A and B independently for its
configured repetition count and writes B's arithmetic mean plus aggregate Result; same-allocation
ABBA writes B's geometric mean plus its paired aggregate Result. Both replacements happen before
the Attempt completes. Failed infrastructure never fabricates an outcome.

Gateway Control schema 6 retains all evaluation records and the authoritative outcome's source
evaluation identity. Artifact liveness includes every recorded candidate and raw result. Authenticated
administration API and CLI operations list records and return the exact candidate files and raw
Gateway result by evaluation ID.

## Consequences

An Attempt may safely explore, fail, repair, and evaluate again without poisoning its final state.
The complete Kernel/result history is inspectable independently of lifecycle Event retention.
Both retention methods reuse their mandatory Candidate-side measurements as final authority,
avoiding a duplicate standalone Candidate Eval while retaining exact comparison evidence. Bootstrap
still spends an independent final Eval because no incumbent exists. If Runtime crashes after
external submission but before persisting the final record, idempotent recovery remains required;
target-deployment crash testing remains required.
