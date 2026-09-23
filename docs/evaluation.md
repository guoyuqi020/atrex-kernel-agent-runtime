# Evaluation and promotion

English | [中文](evaluation.zh.md)

Runtime owns evaluation policy and promotion. Core may request exploratory operations, but Agate
owns GPU execution and Runtime alone decides whether a Kernel or Agent revision is retained.

## Evaluation inputs and privacy

Each Campaign seals one private Evaluation Contract containing the reference implementation, input
generator, validation Shapes, metadata, optional Roofline, tolerances, sampling policy, clock
policy, and Production Gate flag. Runtime replaces Gate-owned fields with deployment policy before
sealing the Contract.

Before sealing, Runtime lexicographically sorts the IDs from complete `shape_valid.json` (or
Contract `shapes`), then shuffles them with a local `random.Random(42)` and divides the shuffled
population into halves (odd extra: Valid). Using that same RNG, it independently samples up to
15 Shapes from each half. Valid gets `min(15, ceil(N/2))` Shapes and Test gets
`min(15, floor(N/2))`; fewer than two Shapes is an error. Extra Shapes do not participate in
evaluation. This RNG does not modify global random state or evaluator correctness seeds. The private Contract
retains only the selected Valid + Test Shapes (at most 30) and seals `validation_shape_ids`;
Test is its complement. Per-Shape metadata and Roofline are reduced to the selected population.
The same partition is used across DSLs, Attempts, retries, and ablation arms.

The private Contract's `shape_split` archives the algorithm, seed, cap, original population count
and IDs, final Valid/Test IDs, and the stable opaque Agent-ID mapping. For an original population
`"0"` through `"9"`, the record is:

```json
{
  "algorithm": "python_random_shuffle_sample",
  "seed": 42,
  "max_shapes_per_set": 15,
  "source_shape_count": 10,
  "source_shape_ids": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  "valid_shape_ids": ["2", "3", "5", "7", "8"],
  "test_shape_ids": ["0", "1", "4", "6", "9"],
  "agent_shape_id_map": {"0": "2", "1": "3", "2": "5", "3": "7", "4": "8"}
}
```

The Campaign's `evaluation_contract_digest` locates the immutable archive at
`<artifacts_root>/sha256/<digest_without_prefix>/payload/value.json`, under `shape_split`.
That Contract also contains the exact selected Shape records, enabling replay without resampling.
This archive is administrative data: it is removed from Agent contexts and per-batch requests,
and must not be copied into Agent workspaces or Evidence.

Agent-facing Valid Shapes are re-keyed to contiguous opaque IDs `"0"` through `"V-1"`. Runtime
uses the private map when building Agent Evaluate/Profile requests and when projecting authoritative
results back into historical Evidence. Original Valid IDs therefore cannot reveal Test membership
through gaps in a dense source-ID sequence. The aliases remain stable for the Campaign.

Agent operations, including ordinary Evaluate, Agent ABBA, Profile, and Check, use Valid only.
Bootstrap final Evaluate, Lineage seed Evaluate, and ordinary Evaluate comparisons also use
Valid. Authoritative Runtime ABBA executes Valid + Test but derives correctness, latency, and
promotion exclusively from Valid. Test is a private, observation-only generalization measurement;
its failure or slowdown cannot reject a Kernel or Agent revision. Bootstrap similarly gates v0 on
Valid and records one non-blocking private Test observation. Custom Agent probes still use
Agent-supplied inputs; they cannot select or reveal hidden Test cases.

Agent-facing historical reports, Evolver summaries, and Attempt fact indexes expose only Valid
per-Shape timings and recompute their latency aggregates from Valid. They never expose full-set
aggregate latencies, Test profiler data, or Test error metrics. Acceptance/selection verdicts are
Valid-only. Private Gateway Results retain Test observations for administration and research;
Result Artifacts remain Agent-visible Valid projections.
Public `shape_train` describes the legal domain, not the holdout membership. Auto-generated
problem context and missing Roofline construction use Valid inputs only.

New Campaigns seal this partition. A pre-change Campaign's immutable Contract is not rewritten:
create a new Campaign/workspace if its Contract has no fixed-seed split archive or no opaque
Agent-ID map; do not mix its old full-set or source-ID results with new Valid-only measurements.
The shared VecAdd example has two Shapes for this reason.

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
the exact Candidate against the trusted Contract's Valid subset. This precheck is not the
authoritative same-allocation ABBA retention decision; that decision also uses Valid while Test is
recorded only as a private observation.
That precheck may come from this Attempt or from an explicit `adopt` Experiment referencing a
compatible successful full Evaluate in visible history. Runtime verifies the original Trial and
exact Kernel/Result binding; adoption neither creates a new measurement nor changes its ownership.
The configured independent retention comparison is unchanged. Incompatible historical evidence
requires a new full Evaluate; modifying comments to obtain another digest is unnecessary.

`evaluate` accepts optional `mode`, `input_py`, and `shapes`. `mode` is `full` (the default) or
`correctness_only`; the latter runs correctness checks without performance measurement or automatic
SOL profiling. `input_py` supplies an Agate-compatible Python `_make_inputs` generator, while
`shapes` supplies a non-empty JSON object of Shape records keyed by integer strings. Each record
is an object compatible with that generator. Either component can be overridden independently;
an omitted component continues to come from the sealed Contract. The trusted reference,
tolerances, and gate policy remain in force, and the private inputs are never returned to the Agent.

Core and Kernel Design Agent additionally accept `input_path` and `shapes_path` as workspace-relative UTF-8 files and
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

Custom-input, custom-Shape, and correctness-only results retain Kernel and Result Artifact
identities, and record the effective `mode` and `input_scope` (`custom` when either component is
overridden, otherwise `contract`). Correctness-only results contain no performance measurements.
These calls cannot replace the full trusted-contract evaluation required for `candidate_ready`,
Kernel retention, or Agent promotion. Omit overrides and use `mode: "full"` (or omit `mode`) for that
evaluation; the existing default request and result format remain unchanged.

### Custom input file example

This paired example is for a public vector-add ABI, `Model.forward(left, right)`, with no Model
constructor arguments and two CUDA float32 vectors. Adapt names, dtypes, devices, constructor
arguments, and legal sizes to your task's public ABI; these illustrative cases are not private
validation Shapes. The generator follows the shared [VecAdd input example](../examples/shared/vecadd/reference/input.py).

Save `scratch/custom-input.py`:

```python
import torch


def _make_inputs(num_elements: int) -> dict[str, torch.Tensor]:
    left = torch.randn((num_elements,), device="cuda", dtype=torch.float32)
    return {"left": left, "right": torch.randn_like(left)}
```

Save `scratch/custom-shapes.json`:

```json
{
  "0": {"input_kwargs": {"num_elements": 1024}, "init_kwargs": null},
  "1": {"input_kwargs": {"num_elements": 4097}, "init_kwargs": null}
}
```

The fields have different roles:

- Each numeric string is a local test-case ID, not a tensor dimension or a request to select a
  hidden case. Multiple records define multiple cases; `4097` illustrates a non-aligned length.
- `input_kwargs` is passed to `_make_inputs(**input_kwargs)`. Its keys must match the generator's
  arguments; do not put `num_elements` directly beside `init_kwargs` or encode tensors here.
- The generator returns a dictionary whose keys match `Model.forward` arguments. Here `left`
  and `right` are tensors, not Shape descriptions. Return this dictionary directly, not a tuple
  or an extra `{"kwargs": ...}` wrapper.
- `init_kwargs` supplies `Model(**init_kwargs)` constructor arguments; use `null` or `{}` when
  there are none. It is separate from the input-generator arguments.
- Let the evaluator control random seeds; do not call `torch.manual_seed` inside the generator.

Save `scratch/evaluate-custom.json`, then invoke the session's `gateway-execute` tool:

```json
{
  "operation": "evaluate",
  "mode": "correctness_only",
  "input_path": "scratch/custom-input.py",
  "shapes_path": "scratch/custom-shapes.json"
}
```

```bash
python3 agent/optimizer/src/runtime_tools.py gateway-execute --request scratch/evaluate-custom.json
```

Use the tool path printed in your Session if it differs. Omit `mode` for correctness plus timing;
the same two files also work with the ABBA comparison
below. Usually supply both files together. A Shapes-only override requires keys compatible with
the retained generator, and a generator-only override must accept the retained Shapes' arguments;
neither override exposes that private component. Custom input values must still satisfy the public
ABI, and a passing custom test does not replace full trusted-contract evaluation.

For direct HTTP clients, send the Python file's contents as `input_py` and the parsed JSON object
as `shapes`. File paths are expanded by the Core/KDA tool, not read from the Agent container by the
HTTP endpoint.

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
The response's Kernel Artifact identifies B; its Result Artifact identifies this comparison. The retained Result Artifact
includes A's `baseline_kernel_artifact_digest`, `baseline` and `candidate` correctness/latency
summaries, all `measurements` and the `schedule`, `mode`, and `input_scope`. `speedup` is
A/B latency and `improvement_pct` is (A−B)/A × 100; aggregate latency uses a geometric mean.
Use `result-artifact-read` to retrieve this evidence. Exploratory ABBA does
not create an ordinary Evaluate record. Runtime retains the normalized per-Shape aggregate for both
A and B and keeps underlying responses as private evidence.

## Ordinary Evaluate Shape batches

Each ordinary Evaluate round submits one Agate Eval Job per validation Shape, with at most sixteen
batches in flight. This default matches ABBA's one-Shape/sixteen-batch setting and applies to Optimizer
requests, Bootstrap stages, Lineage seeding, and the ordinary Evaluate comparator. The Agent submits
one logical request; Runtime partitions the sealed contract, including matching metadata and Roofline,
and preserves every batch's Job and result in the aggregate Artifact.

All Shapes must pass. For an Agent full Evaluate, Runtime performs one complete logical Agate call
and combines its per-Shape latencies using the geometric mean. There is no extra cross-job repeat or
median layer. The sixteen-batch cap applies to this call. Bootstrap, Lineage seeding, and trusted
comparators retain their own configured sampling policies. ABBA's comparison settings do not change
the ordinary-Evaluate aggregation.

## Single submission and single measurement

Runtime accepts each exact full ordinary Evaluate or exploratory ABBA task only once within a
Lineage. Task identity covers the exact Candidate Kernel, the Baseline Kernel for ABBA, the
measurement method and parameters, and the sealed input domain. Correctness-only Evaluate, Profile,
Dev, Check, and Disassemble are outside this rule because they do not share the same per-Shape timing
contract.

- The first accepted task executes one logical Agate call, partitioned into Shape batches.
- Runtime validates Shape coverage and mechanically computes aggregate latency and ABBA speedup.
  Explicit correctness failures are retained.
- Runtime returns one Agent-visible Result Artifact with
  `measurement_aggregation: {"repetitions": 1, "method": "single_measurement"}`. Raw responses
  remain private evidence. Inner GPU benchmark sampling and ABBA's configured A/B schedule are unchanged.
- A later Agent invocation of the identical task is rejected before Agate execution. The error names
  `previous_result_artifact_digest` and directs the Agent to `result-artifact-read`; transport retry
  of the original invocation remains idempotent and replays the same response.

This rule prevents an Agent from spending evaluator capacity or choosing among repeated samples by
resubmitting unchanged code. Changing the Kernel, Baseline, input domain, or measurement parameters
creates a different task.

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
Evaluate. Transport failures retry the request; terminal failures explicitly classified as
`error_class=infra` resubmit only the failed batch with a new Job ID, using 5/10/20/40-second
backoff then 60 seconds indefinitely until recovery or cancellation. Completed sibling batches
are retained, without advancing Session recovery generations. This also applies to authoritative
and Agent ABBA. Candidate validation, compilation and correctness failures, unclassified errors,
and cancelled Jobs are not infrastructure retries.

An ordinary Attempt uses `kernel_retention_comparison`:

- `evaluate`: independently measures incumbent A and Candidate B for the configured number of
  repeats; the Candidate must be correct and exceed the configured uncertainty threshold.
- `same_allocation_abba`: runs interleaved A/B measurements inside one Agate allocation per Shape
  batch. Each repeat measures both revisions; pair order alternates between `A, B` and `B, A`, so
  two repeats produce `A, B, B, A`. Runtime executes that complete schedule once, validates Shape
  coverage, and computes the authoritative geomean without a cross-job median. An explicit
  correctness failure fails the comparison. Every physical run remains recorded. Single-file
  Kernels use Agate's native Eval ABBA API; Runtime reconstructs its authoritative ledger from the
  returned raw SDK runs. Multi-file source trees retain the commit-pinned Dev driver because the
  native wire schema carries one source file per side.

For authoritative ABBA, Runtime records each completed physical Shape batch under an identity
covering the exact revision pair, sealed Contract, execution transport, evaluator where applicable,
purpose, schedule, and repetition.
Resuming the same comparison reads completed batches from the Registry and Artifact Store instead
of submitting them again. Transient Agate failures retry the affected batch with a fresh Job, and a
different revision pair starts fresh measurements.

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
