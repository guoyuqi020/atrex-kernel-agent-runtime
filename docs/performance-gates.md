# Performance gates

Kernel retention and Agent promotion are independent Campaign policies. Either policy may use an
ordinary Evaluate comparison or `same_allocation_abba`; selecting one does not change the other.
All paths use the Campaign Contract sealed from Runtime `gate_policy`; task input cannot override
Gate-owned sampling, tolerances, timeouts, clock policy, or evaluator identity.

When `gate_policy.production_gate` is true, a content-level Production Gate precedes correctness and
performance. It enforces the fixed DSL, self-contained source, approved dependencies, and absence of
PyTorch/prebuilt-kernel fallbacks. A policy rejection never reaches Agate and cannot be retained or
promoted even if an older evaluation exists.

## Ordinary Evaluate

`gate_policy.optimizer` controls exploratory Evaluate sampling. The Atrex-compatible policy uses
one correctness case, one Eval, and 100 eager bench iters. Bootstrap does not reuse this sampling:
it executes the ordered `gate_policy.bootstrap.stages`, first one correctness case and then five,
with 5 eager bench iters in each stage. Both stages must pass; only the second stage supplies the
authoritative `v0` latency. Attempt finalization belongs entirely to the selected retention policy.

Runtime runs exactly `repeats` independent logical Eval measurements concurrently for each Kernel,
requires every measurement to be correct, compares arithmetic means, and accepts only an absolute
improvement
greater than `measurement_uncertainty_us`. B's arithmetic mean and a new aggregate Artifact that
references every raw B result become the Candidate Kernel's authoritative Evaluation before the
Attempt completes. This applies even when `repeats: 1`; no earlier exploratory latency is promoted.

Every logical ordinary Evaluate uses one trusted batching executor shared by Optimizer exploration,
Bootstrap stages, Lineage Seed validation, and the ordinary Evaluate Comparator. It deterministically
splits the sealed Contract into groups of four Shapes and runs at most four Shape batches concurrently.
Every Shape must pass. Runtime reconstructs the full-workload latency as the Shape-count-weighted
geometric mean of the batch latencies and retains every physical Agate Job in the aggregate result.
A Contract with at most four Shapes remains one unchanged Agate Job.

## Same-allocation ABBA

For `repeats: 2`, the exact schedule is incumbent, candidate, candidate, incumbent. Larger repeat
counts continue alternating pair order. Runtime exports the evaluator-only subset of the configured
Atrex Bench commit and uploads it with immutable incumbent/candidate source snapshots and the sealed
private Evaluation Contract.

Each Shape batch is one Agate `dev` Job and therefore one GPU allocation. A remote Runtime-owned
driver swaps only the candidate source and launches a fresh canonical Atrex Bench evaluation for
every schedule entry. Separate Shape batches may run concurrently on separate allocations. Runtime
accepts no partial or reordered schedule and never combines runs from a failed allocation retry.
`gate_policy.lock_clocks` defaults to `true`. When it is true, the remote driver locks the allocation once before
the first A/B entry, exports Atrex Bench's external-lock markers to every evaluator subprocess, and
resets clocks after the complete schedule. Failure to apply or reset the requested lock fails the
batch closed. `lock_clocks: false` performs no clock mutation.

After all batches finish, Runtime reconstructs each full-workload run from its per-Shape latencies,
then computes one geometric mean across repeats for each Kernel. All incumbent and candidate runs
must pass compile, correctness, and performance. The candidate wins only when:

```text
100 * (incumbent_geomean - candidate_geomean) / incumbent_geomean
    > minimum_improvement_percent
```

The strict `>` retains the incumbent on an exact threshold or tie. Every reconstructed A/B run is
stored in `kernel_measurements`; all runs reference one aggregate Gateway Result artifact containing
the raw batch jobs, requested schedule, parsed remote payloads, sealed Evaluation Contract digest,
and authoritative Incumbent/Candidate latency and Roofline SOL aggregates. Each remote canonical
evaluation returns per-Shape `sol.pct`; Runtime preserves it and geometrically aggregates the
Candidate values across repeats and Shapes. Runtime events record batch Job IDs, the Atrex Bench
commit, the canonical evaluator Bundle digest, and comparison completion.
For Kernel retention, the Candidate geometric mean and this same aggregate Artifact replace the
provisional exploratory Evaluation on the new Kernel revision before the Attempt completes. No
separate Runtime-final ordinary Eval is submitted. Bootstrap uses its two ordered independent stages
because `v0` has no incumbent for an A/B comparison.

The same-allocation guarantee relies on Agate's Job contract: one running `dev` Job owns one
allocation. Runtime does not independently attest a physical GPU UUID. If hidden Shapes are divided
into multiple batches, the guarantee applies independently within each batch, matching the legacy
Atrex Kernel Agent verifier behavior.
