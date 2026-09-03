"""Dev-pod correctness probe for fused_moe_fp8 candidates (cuda/sm_120).

Runs INSIDE a gateway-execute dev job, beside the candidate kernel.py.
It builds contract-shaped random inputs, runs the candidate Model, and
compares against an independent fp32 dequant-matmul reference that
reproduces the public quantization semantics (per-128-group activation
scales amax/448, e4m3 RNE, 128x128 weight-block scales, fp32 accumulation,
bf16 rounding at SiLU(gate)*up, fp32 top-k weighted reduction).

This probe is NOT the evaluator. Its reference may differ from the
authoritative one in sub-bf16 rounding details, so its threshold is twice
the evaluation tolerance (2*atol, 2*rtol). Failing this probe almost
certainly means the candidate fails evaluation; passing it is necessary,
not sufficient.

Invocation (gateway-execute request):
{
  "operation": "dev",
  "command": "python3 moe_check_probe.py [token_count]",
  "file_paths": ["tools/moe_check_probe.py"],
  "intent": "custom_harness"
}
"""

import sys

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


def quantize_groups(x: torch.Tensor) -> torch.Tensor:
    """Per-128-group fp8 e4m3 quantize/dequantize; x [..., K] fp32."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1] // GROUP, GROUP)
    scale = torch.clamp(flat.abs().amax(dim=-1, keepdim=True), min=1e-12) / FP8_MAX
    q = (flat / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.to(torch.float32) * scale).reshape(shape)


def dequant_weight(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """w_fp8 [N, K] e4m3, scale [N/128, K/128] fp32 -> fp32 [N, K]."""
    n, k = w_fp8.shape
    w = w_fp8.to(torch.float32)
    s = scale.repeat_interleave(GROUP, dim=0).repeat_interleave(GROUP, dim=1)
    return w * s


def reference(hidden, w1, w2, tw, ids, w1s, w2s):
    t = hidden.shape[0]
    hd = quantize_groups(hidden.float())
    out = torch.zeros(t, HIDDEN, dtype=torch.float32, device=hidden.device)
    flat_ids = ids.reshape(-1)
    weights = tw.reshape(-1)
    rows = hd.repeat_interleave(TOP_K, dim=0)  # [T*8, K]
    for e in range(NUM_EXPERTS):
        sel = (flat_ids == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        w1dq = dequant_weight(w1[e], w1s[e])  # [N1, K]
        w2dq = dequant_weight(w2[e], w2s[e])  # [N2, I]
        g1 = rows[sel] @ w1dq.t()  # fp32 [m, 1024]
        gate = g1[:, :INTER].to(torch.bfloat16)
        up = g1[:, INTER:].to(torch.bfloat16)
        inter = (F.silu(gate) * up).float()
        iq = quantize_groups(inter)
        o2 = iq @ w2dq.t()  # fp32 [m, 2048]
        wcol = weights[sel].unsqueeze(1)
        t_idx = sel // TOP_K
        out.index_add_(0, t_idx, wcol * o2)
    return out.to(torch.bfloat16)


def main():
    torch.manual_seed(0)
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
    logits = torch.randn(t, NUM_EXPERTS, device=dev)
    ids = torch.topk(logits, TOP_K, dim=1).indices.to(torch.int32)
    tw = torch.rand(t, TOP_K, device=dev) + 0.1
    tw = tw / tw.sum(dim=1, keepdim=True)
    tw = tw.to(torch.float32)

    from kernel import Model

    model = Model(num_experts=NUM_EXPERTS, intermediate_size=INTER,
                  top_k=TOP_K, block_shape=[GROUP, GROUP])
    got = model(hidden, w1, w2, tw, ids, w1s, w2s)
    torch.cuda.synchronize()
    ref = reference(hidden, w1, w2, tw, ids, w1s, w2s)

    got32 = got.float()
    ref32 = ref.float()
    abs_err = (got32 - ref32).abs()
    tol = 2 * ATOL + 2 * RTOL * ref32.abs()
    bad = (abs_err > tol).sum().item()
    denom = ref32.abs().clamp_min(1e-6)
    print("PROBE",
          "tokens=%d" % t,
          "max_abs_err=%.6g" % abs_err.max().item(),
          "max_rel_err=%.6g" % (abs_err / denom).max().item(),
          "bad_elements=%d" % bad,
          "shape_ok=%s" % (tuple(got.shape) == (t, HIDDEN)),
          "dtype_ok=%s" % (got.dtype == torch.bfloat16))
    print("PROBE_VERDICT", "PASS" if bad == 0 else "FAIL")


if __name__ == "__main__":
    main()
