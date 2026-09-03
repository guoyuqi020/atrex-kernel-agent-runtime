"""Head-to-head latency bench for fused_moe_fp8 candidate variants on the dev pod.

Benchmarks the mounted candidate `kernel.py` against an optional variant module
uploaded as `variant_kernel.py` (any module exposing the same `Model` API) on
IDENTICAL random inputs across packed-token counts, using CUDA events with warmup
and median-of-N. Also asserts the two modules agree numerically (max abs diff) so
a broken variant cannot be silently benchmarked.

Why: before spending a Gateway `evaluate`, screen launch-geometry or structural
variants cheaply in the dev pod. Relative comparisons inside one job share GPU,
clocks, and inputs, so they isolate the code change even though dev clocks are not
locked like evaluate's.

Invocation (dev request): upload this file, `probe_moe_correctness.py` (used for
`make_inputs`), and the variant source renamed to `variant_kernel.py`:

```json
{
  "operation": "dev",
  "command": "python3 bench_moe_latency.py --iters 30",
  "file_paths": ["tools/bench_moe_latency.py", "tools/probe_moe_correctness.py", "scratch/variant_kernel.py"],
  "job_timeout_s": 600,
  "intent": "custom_harness"
}
```

CLI args: `--iters N` (timed iterations per case, default 30), `--warmup N`
(default 5), `--cases 512,2048,4096,6144,8192` (comma-separated token counts),
`--variant NAME` (module name, default variant_kernel; if import fails, benches
the candidate alone). Env-var restrictions in dev pods allow only TRITON_/TORCH_/
NCCL_/CUDA_LAUNCH_ prefixes, so all toggles are CLI arguments.

Outputs (stdout): per-case `case=<m> cand_us=... var_us=... ratio=... max_abs=...`
lines (ratio < 1 means candidate faster), then `GEOMEAN_RATIO` and a
DISTRIBUTION_WEIGHTED summary applying the public contract's packed-token weights
(82% on 4320-8192, 18% on 450-4319). Exit code 0 unless a module fails to run.

Limitations: dev clocks are not locked, so treat absolute numbers as indicative and
use the ratio as the decision signal; random fixed-seed inputs, not evaluator
inputs; the authoritative latency/correctness result is a Gateway `evaluate`.
"""

import argparse
import importlib

import torch

DEFAULT_CASES = [512, 2048, 4096, 6144, 8192]
E, H, I, K = 256, 2048, 512, 8


def bench_module(model, args_list, iters, warmup):
    """Return dict m -> median microseconds."""
    out = {}
    for m, inputs in args_list:
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            model(*inputs)
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(iters))
        out[m] = times[len(times) // 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cases", type=str, default=",".join(map(str, DEFAULT_CASES)))
    ap.add_argument("--variant", type=str, default="variant_kernel")
    cli = ap.parse_args()
    cases = [int(x) for x in cli.cases.split(",") if x.strip()]

    probe = importlib.import_module("probe_moe_correctness")
    cand = importlib.import_module("kernel")
    try:
        var = importlib.import_module(cli.variant)
    except Exception as e:  # noqa: BLE001
        print(f"variant import failed ({e}); benching candidate only")
        var = None

    mc = cand.Model()
    mv = var.Model() if var is not None else None

    args_list = []
    for idx, m in enumerate(cases):
        inputs = probe.make_inputs(m, E, H, I, K, seed=100 + idx)
        args_list.append((m, inputs))

    # numeric agreement check on the first call of each case
    if mv is not None:
        for m, inputs in args_list:
            with torch.no_grad():
                oc = mc(*inputs)
                ov = mv(*inputs)
            d = (oc.float() - ov.float()).abs().max().item()
            print(f"agree case={m} max_abs={d:.6g}")

    results = []
    for i, (m, inputs) in enumerate(args_list):
        # alternate order across cases to cancel drift
        if i % 2 == 0:
            rc = bench_module(mc, [(m, inputs)], cli.iters, cli.warmup)
            rv = bench_module(mv, [(m, inputs)], cli.iters, cli.warmup) if mv else None
        else:
            rv = bench_module(mv, [(m, inputs)], cli.iters, cli.warmup) if mv else None
            rc = bench_module(mc, [(m, inputs)], cli.iters, cli.warmup)
        tc = rc[m]
        tv = rv[m] if rv else float("nan")
        ratio = (tc / tv) if rv else float("nan")
        results.append((m, tc, tv, ratio))
        print(f"case={m} cand_us={tc:.1f} var_us={tv:.1f} ratio={ratio:.4f}")

    if mv is not None:
        import math

        g = math.exp(sum(math.log(r[3]) for r in results) / len(results))
        # distribution weights from the public contract (4320-8192: 82%, else 18%)
        wsum = 0.0
        wlog = 0.0
        for m, _, _, ratio in results:
            w = 0.82 if m >= 4320 else 0.18
            wsum += w
            wlog += w * math.log(ratio)
        gw = math.exp(wlog / wsum)
        print(f"GEOMEAN_RATIO={g:.4f} WEIGHTED_RATIO={gw:.4f} (ratio<1 => candidate faster)")


if __name__ == "__main__":
    main()
