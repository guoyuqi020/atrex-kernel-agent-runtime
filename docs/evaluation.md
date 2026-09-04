# Evaluation and promotion

English | [中文](evaluation.zh.md)

Runtime owns evaluation policy and promotion. Core may request exploratory operations, but Agate
owns GPU execution and Runtime alone decides whether a Kernel or Agent revision is retained.

## Evaluation inputs and privacy

Each Campaign seals one private Evaluation Contract containing the reference implementation, input
generator, validation Shapes, metadata, optional Roofline, tolerances, sampling policy, clock
policy, and Production Gate flag. Runtime replaces Gate-owned fields with deployment policy before
sealing the Contract.

Agents never receive exact validation Shapes, `reference.py`, `input.py`, metadata, or Roofline.
They receive a public `shape_train` contract describing the legal parameter domain and non-Shape
ABI constraints. Gateway responses expose only aggregate correctness, aggregate latency, latency
by opaque numeric Shape ID, and sanitized profiler data.

## Exploratory operations

Optimizer Runtime Tools expose `check`, `dev`, `evaluate`, `profile`, `disassemble`, `poll`, `jobs`,
`cancel`, `env`, `health`, and `config` through `gateway-execute`. Candidate-bearing operations seal
the exact source before calling Agate. Every result is immutable and queryable by the identities
returned to the Agent.

An exploratory `evaluate` is authoritative measurement evidence, but it does not create a `vN`
Kernel revision. The Agent may evaluate several Candidates in one Attempt, record them in the
Experiment Journal, and nominate one evaluated Candidate in `attempt-report`.

## Correctness and Production Gate

`gate_policy` defines tolerances, correctness-case counts, warmup/benchmark budgets, timeouts,
clock locking, and the pinned Atrex Bench evaluator. Bootstrap, Optimizer exploration, retention,
Agent promotion, and Lineage seeding use the same sealed policy with role-specific sampling.

When `production_gate` is enabled, Runtime applies a trusted source-content check before GPU
execution and again before publication. It enforces the selected DSL, rejects PyTorch compute
fallbacks and dynamic/prebuilt implementation loading, and validates `solution.json` when present.
Exploratory operations report safe Production Gate warnings so the Agent can repair a Candidate;
publication still fails closed.

## Bootstrap and Kernel retention

Bootstrap runs ordered correctness stages from `gate_policy.bootstrap`. A successful terminal
Candidate becomes Lineage-local Kernel `v0`; there is no incumbent comparison. Because that `v0` is
the absolute baseline every later Kernel revision is reported against,
`gate_policy.bootstrap.bench_iters` must sample as deeply as the Optimizer and Retention gates — a
shallower budget shifts the same Kernel's latency by tens of percent on slow operators.

An ordinary Attempt uses `kernel_retention_comparison`:

- `evaluate`: independently measures incumbent A and Candidate B for the configured number of
  repeats; the Candidate must be correct and exceed the configured uncertainty threshold.
- `same_allocation_abba`: runs interleaved A/B measurements inside one Agate allocation per Shape
  batch. Each repeat measures both revisions; pair order alternates between `A, B` and `B, A`, so
  two repeats produce `A, B, B, A`. Runtime validates the schedule and stores every run.

The selected comparator's B aggregate is the Candidate's authoritative latency. There is no second
independent Attempt-final evaluation after comparison. An Attempt that produces no valid
nomination remains in Attempt history without consuming a Kernel version.

Agent promotion uses `agent_promotion_comparison` independently from Kernel retention. The best
Kernel produced by each competing Agent participates; Runtime may retain a Kernel without
promoting its Agent, or promote a Challenger after the configured comparison succeeds.

## Roofline and SOL

Resolution order is:

1. preserve an explicit Evaluation Contract Roofline;
2. reuse the Campaign-sealed Roofline on resume;
3. if configured, execute the commit-pinned Atrex Bench Roofline builder and validate exact Shape
   coverage;
4. when the sealed Contract has no Roofline, run an NCU SOL Profile after each correct Evaluate.

The builder runs trusted code from one full Atrex Bench commit with bounded input/output and no
Agent authority. Generated output is sealed into the Campaign Contract before Agent execution.
Profile failure does not invalidate correctness or latency; SOL remains unavailable.

Automatic NCU fallback is selected only when the sealed Contract's `roofline` field is null. A
structurally valid explicit Roofline that lacks the actual Agate device key can therefore produce
no SOL and does not trigger automatic fallback. Operators should generate a device-compatible
Roofline or omit it.

Kernel catalogs report all-Shape SOL as a geometric mean when every Shape supplies a value;
otherwise JSON uses `null` and tables show `-`.

## Durable evidence

Runtime retains exact candidate source, raw Agate result, normalized measurements, comparator runs,
Kernel Trials, Attempt reports, and versioned Kernel/Agent outcomes. Worker projections remain
privacy-preserving; administration interfaces can retrieve bounded exact source and raw results.
See [Interface Reference](interfaces.md) for commands and [Protocols](protocols.md) for identity and
visibility semantics.
