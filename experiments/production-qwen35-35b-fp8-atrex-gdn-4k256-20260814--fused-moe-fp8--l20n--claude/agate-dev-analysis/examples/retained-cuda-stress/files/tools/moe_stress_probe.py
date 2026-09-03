"""Dev-pod stress probe: skewed routing, out-of-domain ids, and timing.

Usage: python3 moe_stress_probe.py [token_count]
Prints SKEW/OOD/TIMING lines and a final STRESS_VERDICT PASS|FAIL.
"""
import sys
import time

import torch
import torch.nn.functional as F

NUM_EXPERTS = 256
HIDDEN = 2048
INTER = 512
TOP_K = 8
GROUP = 128
FP8_MAX = 448.0
ATOL = 0.01
RTOL = 0.05


def quantize_groups(x):
    shape = x.shape
    flat = x.reshape(-1, shape[-1] // GROUP, GROUP)
    scale = torch.clamp(flat.abs().amax(dim=-1, keepdim=True), min=1e-12) / FP8_MAX
    q = (flat / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.to(torch.float32) * scale).reshape(shape)


def dequant_weight(w_fp8, scale):
    w = w_fp8.to(torch.float32)
    s = scale.repeat_interleave(GROUP, dim=0).repeat_interleave(GROUP, dim=1)
    return w * s


def reference(hidden, w1, w2, tw, ids, w1s, w2s):
    t = hidden.shape[0]
    hd = quantize_groups(hidden.float())
    out = torch.zeros(t, HIDDEN, dtype=torch.float32, device=hidden.device)
    flat_ids = ids.reshape(-1)
    weights = tw.reshape(-1)
    rows = hd.repeat_interleave(TOP_K, dim=0)
    for e in range(NUM_EXPERTS):
        sel = (flat_ids == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        w1dq = dequant_weight(w1[e], w1s[e])
        w2dq = dequant_weight(w2[e], w2s[e])
        g1 = rows[sel] @ w1dq.t()
        gate = g1[:, :INTER].to(torch.bfloat16)
        up = g1[:, INTER:].to(torch.bfloat16)
        inter = (F.silu(gate) * up).float()
        iq = quantize_groups(inter)
        o2 = iq @ w2dq.t()
        wcol = weights[sel].unsqueeze(1)
        t_idx = sel // TOP_K
        out.index_add_(0, t_idx, wcol * o2)
    return out.to(torch.bfloat16)


def check(got, ref, label):
    got32 = got.float()
    ref32 = ref.float()
    abs_err = (got32 - ref32).abs()
    tol = 2 * ATOL + 2 * RTOL * ref32.abs()
    bad = (abs_err > tol).sum().item()
    print("%s max_abs_err=%.6g bad=%d shape_ok=%s dtype_ok=%s" % (
        label, abs_err.max().item(), bad,
        tuple(got.shape) == tuple(ref.shape), got.dtype == torch.bfloat16))
    return bad == 0


def main():
    torch.manual_seed(1)
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    dev = torch.device("cuda", 0)

    hidden = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device=dev)
    w1 = (torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN, device=dev) / 16.0).to(
        torch.float8_e4m3fn)
    w2 = (torch.randn(NUM_EXPERTS, HIDDEN, INTER, device=dev) / 16.0).to(
        torch.float8_e4m3fn)
    w1s = torch.rand(NUM_EXPERTS, 2 * INTER // GROUP, HIDDEN // GROUP,
                     dtype=torch.float32, device=dev) * 0.02 + 0.005
    w2s = torch.rand(NUM_EXPERTS, HIDDEN // GROUP, INTER // GROUP,
                     dtype=torch.float32, device=dev) * 0.02 + 0.005

    from kernel import Model

    model = Model(num_experts=NUM_EXPERTS, intermediate_size=INTER,
                  top_k=TOP_K, block_shape=[GROUP, GROUP])
    ok = True

    # 1. All tokens routed to a single expert (max skew, large per-expert M).
    tw = torch.rand(t, TOP_K, device=dev) + 0.1
    tw = (tw / tw.sum(dim=1, keepdim=True)).to(torch.float32)
    ids = torch.zeros(t, TOP_K, dtype=torch.int32, device=dev)
    got = model(hidden, w1, w2, tw, ids, w1s, w2s)
    torch.cuda.synchronize()
    ok &= check(got, reference(hidden, w1, w2, tw, ids, w1s, w2s), "SKEW1")

    # 2. Two experts, odd row counts (partial tiles).
    ids2 = torch.zeros(t, TOP_K, dtype=torch.int32, device=dev)
    ids2[::2, :] = 7
    ids2[:, 3] = 255
    got = model(hidden, w1, w2, tw, ids2, w1s, w2s)
    torch.cuda.synchronize()
    ok &= check(got, reference(hidden, w1, w2, tw, ids2, w1s, w2s), "SKEW2")

    # 3. Out-of-domain ids (-1 and 300) must contribute zero.
    ids3 = torch.full((t, TOP_K), -1, dtype=torch.int32, device=dev)
    ids3[:, 0] = 300
    ids3[:, 1] = 5
    got = model(hidden, w1, w2, tw, ids3, w1s, w2s)
    torch.cuda.synchronize()
    ok &= check(got, reference(hidden, w1, w2, tw, ids3, w1s, w2s), "OOD")

    # 4. Timing: 20 iterations after warmup, CUDA-event timed.
    logits = torch.randn(t, NUM_EXPERTS, device=dev)
    ids4 = torch.topk(logits, TOP_K, dim=1).indices.to(torch.int32)
    for _ in range(3):
        model(hidden, w1, w2, tw, ids4, w1s, w2s)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    n = 20
    for _ in range(n):
        model(hidden, w1, w2, tw, ids4, w1s, w2s)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n
    print("TIMING tokens=%d avg_ms=%.3f" % (t, ms))

    print("STRESS_VERDICT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
