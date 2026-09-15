# FA4 supplied-asset alignment audit

English | [中文](ALIGNMENT.zh.md)

Sources: `atrex-bench-new-qwen38-fa4-gdn` and `fa4-prefill-aka-r0-startpoint-20260915`.
Assets and the formal operator evaluation definition match the supplied packages. Runtime
scheduling and measurement policy are not identical to the original smoke checks.

## Exact asset checks

- Reference and input generator match both the new Benchmark and R0 package byte-for-byte.
- Shape Train, all 30 Shape Valid entries, Metadata and Roofline match the Benchmark bytes.
- Adapter matches R0 `kernel.py`; only its packaged filename changes.
- All 70 Source Bundle files match R0, without additions, removals or modifications. Runtime
  adds the fixed adapter for a total of 71 Candidate files.
- All 32 Evaluator Bundle files match the supplied Benchmark. Runtime exports 31 evaluator
  code files; README is not uploaded.
- The original two smoke scripts, legacy Shape document, provenance and historical validation
  records are preserved byte-for-byte.

`asset-integrity.json` records original-file and Bundle hashes. Preparation checks hashes,
complete source file sets and Commits before publishing a workspace. Normalizing R0's legacy
list-shaped Shape document by ID gives exactly the new dictionary-shaped validation contract,
including every `init_kwargs` and `input_kwargs` entry.

The generator is unchanged: BF16 normal Query samples scaled by 1.22 and KV by 0.96, converted
to FP8 E4M3; zero-initialized workspace; original page tables, sequence lengths, scales and output
allocation. No smaller synthetic workload substitutes for the supplied shapes. Private OSS raw
tensors have not been downloaded; the supplied formal evaluation also uses this synthetic generator.
Benchmark `solution.py` is not the Candidate seed: the requested FA4 R0 remains the starting point,
without optimized C05/Increment source.

## Smoke versus formal evaluation

The upstream P128 smoke checks a fixed P128 example for output shape and finite values. Target
smoke runs one selected shape (default 0), seed 1, and accepts relative L2 <= 0.05. It does not
perform formal input-side-effect validation and invokes Candidate/Reference with the same input
objects. Neither smoke measures latency or registers `v0`.

Formal evaluation uses the unchanged supplied evaluator over all 30 private Shapes, with
elementwise atol=0.06, rtol=0.04 for returned output and mutated `out`. It preserves scratch/mutation
declarations and deterministic Benchmark stage/Shape/case seeds, rather than replacing them with
seed 1. The public training domain never substitutes for validation Shapes. The new `smoke` runner
exists only to reproduce the original commands from temporary pristine R0 inputs; it is not an
acceptance gate or a test of an Agent-modified Candidate.

## Caller-owned measurement differences

The supplied Benchmark owns input and correctness policy, while caller settings own case counts,
warmup, timeouts and ABBA. Compared with bare `run_eval.py` defaults:

| Parameter | Supplied defaults | Runtime task |
| --- | --- | --- |
| Mode | eager | eager |
| Warmup / benchmark | 10ms / 100ms | 10ms / 100ms, despite legacy `*_iters` names |
| Correctness cases | 1 per Shape | Bootstrap 1 then 5; Optimizer 5; Retention 1 |
| Candidate / performance timeout | 60s / 600s | Candidate 120s; ordinary tree performance and outer run budget 600s; each ABBA A/B run budget 120s |
| Clock policy | off unless configured | locked externally, evaluator checks external marker |
| Batching/repetition | given Shapes in one invocation | 1 Shape per batch, up to 16 concurrent batches; one logical ordinary Evaluate; ABBA three complete comparisons with per-Shape medians, repeats=2 (A/B/B/A) each |

Thus evaluator code and acceptance definition match, but complete execution conditions do not.
Hash equality is not evidence of equal latency across policies or measurement windows.

Currently `performance_timeout_seconds=120` does not override ordinary source-tree performance:
the reused Driver Builder uses `evaluation_timeout_seconds=600`. This audit records the actual
transport rather than assuming the configured value is applied; a transport regression test covers it.

In this deployment Agate resource `L20D` denotes B300. Preserve the supplied
`NVIDIA B300 (SM100)` Roofline values; Runtime transport only strips the parenthesized suffix.
Actual SOL/NCU availability still requires remote validation.

This audit starts no models, services or GPU jobs. The supplied validation record remains historical,
not remeasured. Source-tree adaptation of the static Production Gate is not implemented by this asset
audit; that configuration remains off.
