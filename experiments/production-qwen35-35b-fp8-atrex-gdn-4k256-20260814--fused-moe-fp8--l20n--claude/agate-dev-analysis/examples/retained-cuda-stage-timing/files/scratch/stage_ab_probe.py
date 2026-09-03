"""Dev-pod per-stage timing A/B for fused_moe_fp8 candidates (cuda/sm_120).

Runs INSIDE a gateway-execute dev job, beside the candidate kernel.py and a
baseline build uploaded as baseline_kernel.py. For each module it disables
CUDA-graph replay (module-level toggle), warms up, then wraps the driver's
cuLaunchKernel with per-launch CUDA events to obtain a per-stage timeline of
the 8-launch pipeline (plain-launch path only). Prints one STAGEAB line per
module/stage with the median over N repeats, plus the total event-timed
forward median. Used to attribute a kernel edit's effect to the edited stage
under identical pod/clock conditions.

Invocation:
{
  "operation": "dev",
  "command": "python3 stage_ab_probe.py [token_count ...]",
  "file_paths": ["scratch/stage_ab_probe.py", "scratch/baseline_kernel.py"],
  "intent": "custom_harness"
}

Limitations: stage intervals include host-prep bubbles between launches
(dynamic clocks, plain-launch path - NOT the graph-replay path the evaluator
uses); treat numbers as same-pod A/B evidence, not absolute kernel durations.
"""

import sys

import torch

NUM_EXPERTS = 256
HIDDEN = 2048
INTER = 512
TOP_K = 8
GROUP = 128

STAGE_NAMES = ("k_hist", "k_scan_desc", "k_scatter", "k_quant_act",
               "gemm1", "k_silu_quant", "gemm2", "k_reduce")


def build_inputs(t, dev):
    hidden = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device=dev)
    w1 = (torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN, device=dev) / 16.0).to(
        torch.float8_e4m3fn)
    w2 = (torch.randn(NUM_EXPERTS, HIDDEN, INTER, device=dev) / 16.0).to(
        torch.float8_e4m3fn)
    w1s = torch.rand(NUM_EXPERTS, 2 * INTER // GROUP, HIDDEN // GROUP,
                     dtype=torch.float32, device=dev) * 0.02 + 0.005
    w2s = torch.rand(NUM_EXPERTS, HIDDEN // GROUP, INTER // GROUP,
                     dtype=torch.float32, device=dev) * 0.02 + 0.005
    logits = torch.randn(t, NUM_EXPERTS, device=dev)
    ids = torch.topk(logits, TOP_K, dim=1).indices.to(torch.int32)
    tw = torch.rand(t, TOP_K, device=dev) + 0.1
    tw = (tw / tw.sum(dim=1, keepdim=True)).to(torch.float32)
    return hidden, w1, w2, tw, ids, w1s, w2s


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def measure(tag, mod, args, reps):
    # Plain-launch path: the graph toggle is a module-level global read by
    # launch_batch, so flipping it here affects only this module.
    mod._GRAPH_REPLAY_ENABLED = False
    model = mod.Model(num_experts=NUM_EXPERTS, intermediate_size=INTER,
                      top_k=TOP_K, block_shape=[GROUP, GROUP])
    model(*args)
    torch.cuda.synchronize()
    rt = mod._RUNTIME_CACHE[0]
    cu = rt.cu
    orig = cu.cuLaunchKernel

    events = []

    def wrapped(*call_args):
        res = orig(*call_args)
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        events.append(ev)
        return res

    stage_samples = {name: [] for name in STAGE_NAMES}
    total_samples = []
    cu.cuLaunchKernel = wrapped
    try:
        for _ in range(reps):
            events = []
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            model(*args)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_end.record()
            torch.cuda.synchronize()
            if len(events) != 8:
                print("STAGEAB_ERROR tag=%s launches=%d (expected 8)"
                      % (tag, len(events)))
                return False
            prev = ev0
            for name, ev in zip(STAGE_NAMES, events):
                stage_samples[name].append(prev.elapsed_time(ev) * 1000.0)
                prev = ev
            total_samples.append(ev0.elapsed_time(ev_end) * 1000.0)
    finally:
        cu.cuLaunchKernel = orig
    for name in STAGE_NAMES:
        print("STAGEAB tag=%s stage=%s median_us=%.1f"
              % (tag, name, median(stage_samples[name])))
    print("STAGEAB tag=%s stage=TOTAL median_us=%.1f"
          % (tag, median(total_samples)))
    return True


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    counts = [int(a) for a in sys.argv[1:]] or [8192, 4096]
    import baseline_kernel
    import kernel
    ok = True
    for t in counts:
        args = build_inputs(t, dev)
        ok &= measure("baseline_T%d" % t, baseline_kernel, args, 15)
        ok &= measure("candidate_T%d" % t, kernel, args, 15)
    print("STAGEAB_VERDICT", "OK" if ok else "ERROR")


if __name__ == "__main__":
    main()
