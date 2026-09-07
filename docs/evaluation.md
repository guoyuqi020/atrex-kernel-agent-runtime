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

Optimizer Runtime Tools expose `check`, `dev`, `evaluate`, `profile`, `disassemble`, and `env`
through `gateway-execute`. Candidate-bearing operations seal
the exact source before calling Agate. Every result is immutable and queryable by the identities
returned to the Agent.

An exploratory `evaluate` records measurement evidence, but it does not create a `vN` Kernel
revision. The Agent may evaluate several Candidates in one Attempt and record them in the
Experiment Journal. A `candidate_ready` nomination still requires a successful full evaluation of
the exact Candidate against the trusted Evaluation Contract.

`evaluate` accepts optional `mode`, `input_py`, and `shapes`. `mode` is `full` (the default) or
`correctness_only`; the latter runs correctness checks without performance measurement or automatic
SOL profiling. `input_py` supplies an Agate-compatible Python `_make_inputs` generator, while
`shapes` supplies a non-empty JSON object of Shape records keyed by integer strings. Each record
is an object compatible with that generator. Either component can be overridden independently;
an omitted component continues to come from the sealed Contract. The trusted reference,
tolerances, and gate policy remain in force, and the private inputs are never returned to the Agent.

Core additionally accepts `input_path` and `shapes_path` as workspace-relative UTF-8 files and
uploads their contents as `input_py` and `shapes`. The input source limit is 128 KiB and the Shape
file limit is 256 KiB. Files must be regular files under real workspace directories; absolute or
traversal paths, symbolic links, and `.runtime` control paths are rejected. Inline and path forms
of the same component are mutually exclusive. Contents, rather than local file names, determine
the request's idempotency key.

```json
{"operation": "evaluate"}
```

```json
{"operation": "evaluate", "mode": "correctness_only"}
```

```json
{"operation": "evaluate", "mode": "correctness_only", "input_path": "scratch/custom-input.py", "shapes_path": "scratch/custom-shapes.json"}
```

Custom-input, custom-Shape, and correctness-only results retain Kernel Trial and Result Artifact
identities, and record the effective `mode` and `input_scope` (`custom` when either component is
overridden, otherwise `contract`). Correctness-only results contain no performance measurements.
These calls cannot replace the full trusted-contract evaluation required for `candidate_ready`,
Kernel retention, or Agent promotion. Omit overrides and use `mode: "full"` (or omit `mode`) for that
evaluation; the existing default request and result format remain unchanged.

## Exploratory ABBA

`evaluate` accepts optional `candidate_path`, including for ordinary single-Kernel evaluation;
omitting it selects the current `work/kernel` tree. To compare Candidate B against baseline A,
provide `comparison: {method: "abba", baseline_path: "scratch/baseline.py"}`. The baseline path is
required inside `comparison`. Each path is workspace-relative and names a regular `.py` file or a Kernel
Bundle directory. A single Python file is uploaded as `kernel.py`, while directories preserve
relative file names. Absolute/traversal paths, symbolic links, `.runtime` control paths, special
files, and empty Bundles are rejected. Both Bundles are sealed and their contents determine the
request identity; renaming a source without changing its uploaded contents does not create a new
comparison.

```json
{"operation": "evaluate", "comparison": {"method": "abba", "baseline_path": "scratch/baseline.py"}}
```

```json
{"operation": "evaluate", "candidate_path": "scratch/candidate-kernel", "comparison": {"method": "abba", "baseline_path": "scratch/baseline-kernel", "repeats": 2}, "input_path": "scratch/custom-input.py", "shapes_path": "scratch/custom-shapes.json"}
```

`comparison.repeats` defaults to 2 and counts observations per side, producing A, B, B, A. Its accepted range
is 2–20, subject to the schedule fitting Runtime's allocation budget. Each Shape batch measures both
sides within one allocation; different Shape batches may use different allocations. ABBA always
uses `mode: "full"` (normally omitted), and rejects `correctness_only`. Both sides share the
selected input generator and Shapes. The same independent `input_py`/`shapes` overrides and
`input_path`/`shapes_path` file helpers as Evaluate are available; omitted components reuse the
private Contract without exposing it.

ABBA remains exploratory even when it uses the trusted Contract. It neither retains a Kernel nor
promotes an Agent, and cannot replace the successful full trusted-contract Evaluate required for
`candidate_ready`. Runtime's authoritative retention and promotion comparisons remain separate.

The response retains `operation: "evaluate"`; `result.comparison` records `method: "abba"` and
the actual `repeats` count. There is no standalone ABBA operation or top-level repeat parameter.
The response's Kernel Trial and Kernel Artifact identities describe B. The retained Result Artifact
includes A's `baseline_kernel_artifact_digest`, `baseline` and `candidate` correctness/latency
summaries, all `measurements` and the `schedule`, `mode`, and `input_scope`. `speedup` is
A/B latency and `improvement_pct` is (A−B)/A × 100; aggregate latency uses a geometric mean.
Use `kernel-trial-show` and `result-artifact-read` to retrieve this evidence. Exploratory ABBA does
not create ordinary Evaluate records or normalized measurement rows.

## Ordinary Evaluate Shape batches

Each ordinary Evaluate round submits one Agate Eval Job per validation Shape, with at most sixteen
batches in flight. This default matches ABBA's one-Shape/sixteen-batch setting and applies to Optimizer
requests, Bootstrap stages, Lineage seeding, and the ordinary Evaluate comparator. The Agent submits
one logical request; Runtime partitions the sealed contract, including matching metadata and Roofline,
and preserves every batch's Job and result in the aggregate Artifact.

All Shapes must pass. Per-Shape latency is combined using the geometric mean; configured independent
Evaluate repeats still average their round-level latency arithmetically. Repeats run independently,
so the sixteen-batch cap is per round, not a global GPU concurrency limit. ABBA's comparison settings
do not change this ordinary-Evaluate default.

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

The supplied policies use `bootstrap.bench_iters=100`, matching ordinary Optimizer Evaluate.
Both default Bootstrap stages (1 then 5 correctness cases) use this sampling budget. Each stage
uses the same one-Shape/sixteen-concurrent-batch executor and shared Agate retry policy as ordinary
Evaluate. Transport failures retry the request; `logs_unavailable` with a successful backend
resubmits only the failed batch with a new Job ID, using 5/10/20/40-second backoff then 60 seconds
indefinitely. Candidate validation and correctness failures are not infrastructure retries.

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
4. when the sealed Contract has no Roofline, run an NCU SOL Profile after each correct full
   Evaluate. Correctness-only evaluations never trigger automatic profiling.

The builder runs trusted code from one full Atrex Bench commit with bounded input/output and no
Agent authority. Generated output is sealed into the Campaign Contract before Agent execution.
Profile failure does not invalidate correctness or latency; SOL remains unavailable.

For trusted-contract evaluations, automatic NCU fallback is selected only when the sealed
Contract's `roofline` field is null. A structurally valid explicit Roofline that lacks the actual Agate device key can therefore produce
no SOL and does not trigger automatic fallback. Operators should generate a device-compatible
Roofline or omit it.

Custom evaluations omit contract-specific Metadata and Roofline. A correct custom `full`
evaluation can therefore use the same automatic profiling fallback when enabled.

Kernel catalogs report all-Shape SOL as a geometric mean when every Shape supplies a value;
otherwise JSON uses `null` and tables show `-`.

## Durable evidence

Runtime retains exact candidate source, raw Agate result, normalized measurements, comparator runs,
Kernel Trials, Attempt reports, and versioned Kernel/Agent outcomes. Worker projections remain
privacy-preserving; administration interfaces can retrieve bounded exact source and raw results.
See [Interface Reference](interfaces.md) for commands and [Protocols](protocols.md) for identity and
visibility semantics.
