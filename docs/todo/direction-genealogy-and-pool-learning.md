# Direction genealogy and Pool learning

[中文](direction-genealogy-and-pool-learning.zh.md)

## Status

Implemented in the main Runtime and simplified AKA: optional proposal ancestry, visible-reference
validation, append-only persistence, and list/load projection. There is no separate graph export
or generated Evolver genealogy file. See [interfaces](../interfaces.md#direction-genealogy)
for the exact contract and commands. Existing measurement, visibility, and Gate policies are unchanged.
Automatic semantic classification and causal Pool-benefit analysis remain future analytical work;
the implementation records Agent declarations and their provenance, not inferred causality.

## Motivation

Pool experiments show that parallel Trajectories contribute more than independent samples of final
latency. They can:

- explore different Directions from the same Kernel;
- implement the same broad Direction in different ways;
- revisit a Direction that another Trajectory rejected because of a noisy or overly strict local
  criterion;
- port a useful change from a non-selected Trajectory onto the selected Kernel; and
- combine independently measured changes into a new Direction.

The Runtime preserves the relevant Reports, Experiments, Kernel Artifacts, and Result Artifacts, but
the relationship between Directions is usually expressed only in Agent-authored prose. A semantic
retry, reimplementation, correction, port, or combination commonly receives a new Direction ID.
Consequently, the current Registry cannot reliably reconstruct why a new Direction exists or which
prior evidence it uses.

This also obscures causal attribution. A correction performed by the next Attempt in the same
Trajectory demonstrates the value of durable Journals and serial Attempts, while a change recovered
from a sibling Trajectory demonstrates a Pool-specific transfer. These cases should be distinguishable
without rereading complete conversations.

## Observed evidence

This audit predates the genealogy fields. References below to the current or future Runtime describe
the archived experiment and the gap it motivated, not a missing feature in today's interface.

The observations below come from the FlashAttention Pool-3 run. Each DSL used two parallel
Trajectories. Each Trajectory executed three serial Attempts in an Epoch, after which the Runtime
selected the best correct Kernel as the common starting point for both Trajectories in the next
Epoch. Pool-3 did not inherit branch-local adaptive Runtime State, but subsequent Attempts could
still query the Directions, Experiments, Kernel Artifacts, and Result Artifacts persisted by the
Runtime.

The audit covered 90 Pool Attempts in the Registry and the 66 terminal Attempt Reports available for
them. Twenty-four Attempts had no terminal Report, so the Direction semantics below cover only those
66 Reports; Kernel selection and performance claims use authoritative Registry records.

### Aggregate Direction activity

The 66 Reports contained 299 Direction events:

| Event | Count |
|---|---:|
| `propose` | 85 |
| `start` | 100 |
| `complete` | 67 |
| `abandon` | 38 |
| `defer` | 9 |

They referenced 104 distinct Direction IDs, ending in 62 completed, 37 abandoned, and 5 deferred
Directions:

| DSL | Completed | Abandoned | Deferred | Total |
|---|---:|---:|---:|---:|
| CUDA | 26 | 13 | 2 | 41 |
| Triton | 13 | 17 | 0 | 30 |
| CuteDSL | 23 | 7 | 3 | 33 |

Only 5 of the 104 stable Direction IDs spanned more than one Attempt. Most recoveries,
reimplementations, ports, corrections, and combinations received new Direction IDs, leaving their
genealogy only in Agent-authored hypotheses, analyses, and terminal Reports. This is the central
observability gap addressed by this TODO.

### Direction selection

There is strong evidence that Pool improved the choice of Directions. Across the 14 Epochs that
produced an improvement, Trajectory 1 supplied eight winning results and Trajectory 2 supplied six.
Every DSL had at least two wins from Trajectory 2, so no consistently weak Trajectory could have been
removed in advance.

| DSL | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---|---|---|---|---|---|
| CUDA | T1 | T2 | T1 | T2 | T1 |
| Triton | T1 | T1 | T2 | Incumbent retained | T2 |
| CuteDSL | T2 | T1 | T1 | T1 | T2 |

CUDA Epoch 4 is the clearest portfolio example. Both Trajectories started from the same Kernel at
approximately `232.631 µs`:

- T1 first investigated medium-size Shapes. Profiling showed that region already at approximately
  91%–93% DRAM SOL, so it abandoned that target and moved to sub-wave split-KV scheduling for small
  Decode Shapes, obtaining approximately `0.735%` improvement.
- T2 pursued warp specialization for heavy Prefill. Its first implementation improved the aggregate
  by approximately `2.62%` but regressed light Decode Shapes by 9%–15%. It repaired the implementation
  into a dual-Kernel Dispatch that enabled the warp-specialized Kernel only when `splits == 1`,
  `pages_est >= 48`, and `tiles >= 64`. The repaired Candidate improved by approximately `3.93%`
  without the light-Shape regression.

The Runtime selected T2 but retained T1's measured Direction and Artifacts. Pool therefore allowed a
conservative direction and a higher-risk direction to proceed simultaneously: the conservative
branch remained a valid fallback, while its local discovery survived when the riskier branch won.

Stagnation lengths show a similar pattern. The longest non-improving stretch for CUDA and CuteDSL
Pool was two steps, compared with four and six for Isolated Best-of-Two at the same Attempt budget.
Triton shortened from five to three steps, but its final Pool result was approximately 1.1% worse
than Isolated. Pool can therefore reduce the probability of being trapped on one path, but it does
not guarantee a better result under a finite budget.

### Correcting a mistaken feasibility judgment

There is a direct cross-Trajectory example. Triton Epoch 4 T1 tested a deep-shrink host wave override.
Five of six preregistered clauses passed: aggregate performance improved by approximately `0.229%`,
the Decode band improved by approximately `0.305%`, correctness passed, the deep control reached
1.2x, and Prefill remained neutral. The Direction was abandoned because one nominally tied cell
flipped by one timing quantum and violated a strict zero-regression boundary. Its Report already
diagnosed a local Gate artifact rather than an infeasible mechanism.

Epoch 5 T2 found the Candidate in durable history, restored it exactly by Artifact, froze a
noise-tolerant criterion before obtaining a new measurement, and retained the result. Authoritative
latency moved from approximately `273.904 µs` to `273.297 µs`. This demonstrates that a sibling
Trajectory can review a prior infeasibility conclusion. Because the gain was small and changing an
internal criterion can invite post-hoc tuning, the case is stronger evidence of recoverability than
of a large performance benefit; the Runtime Comparator still made the final decision.

A separate CUDA case occurred across serial Attempts in one Trajectory. One Attempt combined two
`fa_combine` changes with a new `_pick_splits` Dispatch and saw hidden dense q-active regressions,
leading it to infer that the combine changes could not ship. The next Attempt recognized that the
combine changes had never been measured without the harmful Dispatch. It restored the Incumbent
Dispatch, ported only windowed register prefetch and generalized combine geometry, passed 90/90
Correctness, and retained an approximately 0.4%–0.7% improvement. This demonstrates durable Journal
and serial-Attempt value, not a Pool-specific effect. A future relationship model must distinguish
the two forms of correction.

### Better implementations of the same Direction

There is direct evidence that independent implementations matter. Both CuteDSL Epoch 1 Trajectories
worked on the same broad Attention direction:

- T1 used a pipeline, head-in-M GQA packing, and split-KV, reaching approximately `288.106 µs`.
- T2 first falsified Host Overhead, then implemented GQA packing, a pipeline, and split-KV with
  different compact/dynamic-start handling. A later wide-M `BLOCK_M=128` variant reached
  approximately `258.874 µs` and won.

Triton Epoch 1 showed the same kind of competition. T1 used grouped split-KV, lean Host Dispatch, and
Regime/KV Gates. T2 used generic split-KV and compact global-token partial buffers. T2's early
implementation reached approximately `550.506 µs`, while T1 quickly reached approximately
`290.885 µs` and eventually won.

Duplicate macro Directions are therefore not necessarily wasted work. They help distinguish “the
mechanism is poor” from “this implementation of the mechanism is poor.” The current Runtime has no
Coordinator assigning distinct Directions; overlap emerges stochastically and can be either a useful
implementation tournament or duplicated cost.

### Combining Directions and creating new ones

This is the strongest Pool-specific evidence in the run.

In CUDA Epoch 4, T1 retained sub-wave split retuning at approximately `0.735%`, while T2 retained dual-
Kernel warp specialization at approximately `3.93%` and won the Epoch. Epoch 5 T1 created an explicit
new Direction that ported the former onto the latter:

1. It first established that their Dispatch regions were largely disjoint: the sub-wave rule applied
   at `tiles <= 27`, while the warp-specialized Gate required `tiles >= 64 && splits == 1`.
2. It ported only the Host Dispatch rule and left the Device Kernel Source unchanged.
3. The new ABBA improved by approximately `0.7286%`, nearly reproducing the sibling's original
   `0.735%` delta.

This is a complete causal chain from two sibling Directions measured separately to a new, multi-parent
combination Direction. It is more than selecting the faster sibling.

CuteDSL Epoch 2 to Epoch 3 also combined sibling work. Epoch 2 selected T1's fp16 partial plus split
economy, while non-selected T2 left three independent gains: reversed work order, batched combine/grid
clamp, and a compact work map. Epoch 3 T1 created three Port Directions and applied them incrementally
to the selected Kernel. Its reported increments were approximately `1.409%`, `0.207%`, and `0.205%`,
and authoritative Epoch latency moved from approximately `251.455 µs` to `247.197 µs`. The combined
gain was smaller than the simple sum, teaching the Agent that some mechanisms overlapped.

Combination can also expose a second-order Direction. After CUDA merged sub-wave split and warp
specialization, the Agent hypothesized that deeper splits increased Partial Traffic and tried 2^-4-
scaled FP16 Partial Buffers. Correctness passed, but Dev measurements showed that conversion
instructions outweighed the bandwidth benefit on 2–4 CTA Combine Grids, so it reverted. The new
Direction did not produce a faster Kernel, but it was a new, measured conclusion caused by the
bottleneck exposed after combining prior work.

### Recovering useful work that was not promoted

One CuteDSL sibling produced `cp.async .cg` L1 bypass plus `COMB_CHUNK=12`, with approximately `3.04%`
ABBA improvement, but its terminal Report/Handoff failed and it was not selected. A later Trajectory
searched the durable Journal for `keep_after` measurements whose Artifacts were absent from the Active
Tree, restored the Candidate by Digest, reran Check/Evaluate, and reduced latency from approximately
`247.197 µs` to `239.798 µs`.

This shows how a failed or non-selected Pool branch can remain a component library for later search.
It depends on the Runtime preserving Artifacts and authoritative measurements rather than only the
Epoch winner.

### Performance and cost boundary

At an equal budget of 30 Optimizer Attempts per DSL, Pool-3 and Isolated Best-of-Two produced:

| DSL | Isolated Best-of-Two | Pool-3 | Pool relative change |
|---|---:|---:|---:|
| CUDA | 233.838 µs | 222.558 µs | -4.8% |
| Triton | 270.268 µs | 273.297 µs | +1.1% |
| CuteDSL | 243.521 µs | 235.593 µs | -3.3% |

Across the three DSLs, Pool used approximately 1,155.654M Provider Tokens, compared with approximately
1,279.457M for Isolated Best-of-Two, an observed reduction of about 9.7%. This is not guaranteed by
the Pool protocol and one run is insufficient for causal attribution. Pool did not consistently
create more finalized Candidates either: CUDA produced 16 versus 20, Triton 9 versus 14, and CuteDSL
18 versus 11. CUDA's improvement therefore cannot be explained simply as “more Candidates.”

Overall, the observed Pool behaves as a Direction tournament, a component library of non-winning
branches, and a cross-Trajectory error-correction loop rather than merely Best-of-Two. Evidence is
strongest for implementation competition and cross-branch combinations. Feasibility corrections
also occur but require protection against post-hoc local Gate changes. Triton's counterexample shows
that broadcasting every three Attempts can prematurely prune Directions that need a longer horizon.

## Implemented model

Declare optional, validated relationships when proposing a new Direction, without modifying the
parent's stable identity or lifecycle. Ancestry is immutable after proposal:

```json
{
  "direction_id": "direction_...",
  "relationship": "combination",
  "derived_from_direction_ids": ["direction_A", "direction_B"],
  "supersedes_direction_id": null,
  "derived_from_experiment_ids": ["experiment_X", "experiment_Y"]
}
```

Initial relationship vocabulary:

- `retry`: repeat the same hypothesis after an execution or infrastructure failure;
- `refinement`: narrow or improve the hypothesis while preserving its main mechanism;
- `reimplementation`: test the same mechanism using a materially different implementation;
- `correction`: revisit a prior feasibility or causal conclusion;
- `port`: apply a measured change to a different Kernel lineage point or sibling result; and
- `combination`: create a new hypothesis from two or more prior Directions.

Relationships must reference durable Registry identities. They must not turn Agent analysis into a
measurement fact: Result Artifacts remain authoritative for observations, while the relationship and
its rationale remain Agent-authored claims.

## Runtime and Agent behavior

- `update-direction` should accept the optional relationship fields and return actionable validation
  errors for missing, invisible, or incompatible references.
- Referenced Directions and Experiments must be visible to the current Lineage under the existing
  evidence policy.
- Relationship records should be append-only. Corrections create a new event rather than rewriting
  historical evidence.
- The Optimizer prompt should explain when to reuse a Direction ID and when to create a derived
  Direction.
- List/load expose declared relationships; follow Experiment IDs to retrieve their measured evidence.
- Evolver uses the existing Reports rather than a separate generated genealogy file.

## Pool analysis enabled by this work

Future analysis can join declared ancestry with existing Attempt provenance and measurements to
study the following questions. These metrics are not generated by the Journal tools:

- Direction diversity between sibling Trajectories;
- independent implementations of the same Direction;
- rejected Directions later corrected or recovered;
- changes ported from a non-selected sibling;
- combinations that reproduce the expected component deltas; and
- improvements attributable to cross-Trajectory transfer versus serial work within one Trajectory.

These metrics must be reported alongside authoritative Kernel measurements. They do not by themselves
prove that a Direction caused a performance change.

## Acceptance criteria

- A Direction can declare zero or more validated parent Directions and Experiments.
- The Registry can reconstruct retry, refinement, reimplementation, correction, port, and combination
  edges without reading conversation text.
- List/load preserve the declared parent Direction and Experiment references.
- Cross-Trajectory versus serial attribution requires joining the existing Attempt provenance.
- Existing Direction records remain readable and valid without migration-authored relationships.
- Tests cover invisible references, cycles where disallowed, duplicate references, retry recovery,
  and multi-parent combinations.

## Non-goals

- Automatically accepting an Agent's causal interpretation as fact.
- Forcing parallel Trajectories to explore different Directions.
- Automatically combining every successful change.
- Replacing Kernel retention, Agent promotion, ABBA, or Production Gate decisions.
