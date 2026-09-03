"""Correctness probe for the block-scaled FP8 fused-MoE candidate kernel.

Runs on the Gateway dev pod next to the candidate `kernel.py`. Builds an
independent PyTorch reference for the public operator contract and compares it
against `kernel.Model(...).forward(...)` for several token counts and routing
regimes. Pure diagnostic: the candidate itself is never modified here.

Reference semantics (public contract):
  * w1 [E, 2*I, H] fp8e4m3fn with per-128x128-block scales w1_scale [E, 2I/128, H/128]
  * w2 [E, H, I]   fp8e4m3fn with per-128x128-block scales w2_scale [E, H/128, I/128]
  * activations are dynamically quantized per 128-element K group:
    scale = max(amax, 1e-12) / 448, RTNE cast to fp8e4m3, dequantized back
  * gate = x @ W_gate^T, up = x @ W_up^T, inter = silu(bf16(gate)) * bf16(up)
    with bf16 rounding at each step, then inter is requantized per 128 group
  * partial = inter_q @ W_down^T, output = sum_k topk_weights[t, k] * partial[t, k]

Optional CLI args:
  --cases small,sparse,dup,odd,fourk,sixk   subset of cases (default all)
  --noquant    also compare the small case against an unquantized-activation
               reference (sensitivity diagnostic)

NOTE: dev-pod env vars are restricted to TRITON_/TORCH_/NCCL_/CUDA_LAUNCH_
prefixes, so all toggles are command-line arguments.
"""

import os
import sys
import traceback


def log(msg):
    print(msg, flush=True)


def quant_act_ref(x):
    """Per-128-group dynamic FP8 quantize+dequantize of activation rows."""
    import torch

    R, H = x.shape
    G = 128
    xf = x.to(torch.float32).view(R, H // G, G)
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-12) / 448.0
    q = (xf / scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)
    return (q * scale).view(R, H)


def dequant_w(w, s):
    """Dequantize [E, N, K] fp8 weights with [E, N/128, K/128] block scales."""
    import torch

    E, N, K = w.shape
    wf = w.to(torch.float32)
    se = s.to(torch.float32).repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    return wf * se


def quant_weight_blocks(W):
    """Quantize fp32 weights into (fp8, scale) with 128x128 block scales."""
    import torch

    E, N, K = W.shape
    wb = W.view(E, N // 128, 128, K // 128, 128)
    amax = wb.abs().amax(dim=(2, 4), keepdim=True)
    scale = torch.clamp(amax, min=1e-12) / 448.0
    q = (wb / scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(E, N, K), scale.squeeze(2).squeeze(3).squeeze(-1).to(torch.float32)


def ref_moe(hidden, w1, w2, topk_w, topk_ids, w1_s, w2_s, quant_act=True):
    import torch
    import torch.nn.functional as F

    M, H = hidden.shape
    K = topk_w.shape[1]
    I = w2.shape[2]
    E = w1.shape[0]
    w1f = dequant_w(w1, w1_s)
    w2f = dequant_w(w2, w2_s)
    ha = quant_act_ref(hidden) if quant_act else hidden.to(torch.float32)
    out = torch.zeros(M, H, dtype=torch.float32, device=hidden.device)
    flat_ids = topk_ids.reshape(-1).long()
    flat_w = topk_w.reshape(-1).to(torch.float32)
    tok = torch.arange(M, device=hidden.device).repeat_interleave(K)
    for e in range(E):
        sel = (flat_ids == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        rows = tok[sel]
        A = ha[rows]
        gate = A @ w1f[e, :I].t()
        up = A @ w1f[e, I:].t()
        gate_b = gate.to(torch.bfloat16)
        up_b = up.to(torch.bfloat16)
        inter = (F.silu(gate_b) * up_b).contiguous()  # bf16 multiply, bf16 result
        iq = quant_act_ref(inter) if quant_act else inter.to(torch.float32)
        part = iq @ w2f[e].t()
        out.index_add_(0, rows, flat_w[sel].unsqueeze(1) * part)
    return out.to(torch.bfloat16)


def make_inputs(M, E, H, I, K, seed, sparse_experts=None, duplicate_slot=False):
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=g)
    W1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.float32, generator=g) / (H ** 0.5)
    W2 = torch.randn(E, H, I, device="cuda", dtype=torch.float32, generator=g) / (I ** 0.5)
    w1, w1_s = quant_weight_blocks(W1)
    w2, w2_s = quant_weight_blocks(W2)
    logits = torch.randn(M, E, device="cuda", dtype=torch.float32, generator=g)
    if sparse_experts is not None:
        logits[:, sparse_experts:] = float("-inf")
    vals, ids = logits.topk(K, dim=1)
    topk_w = torch.softmax(vals, dim=-1).to(torch.float32)
    topk_ids = ids.to(torch.int32)
    if duplicate_slot:
        topk_ids[: max(1, M // 16), 3] = topk_ids[: max(1, M // 16), 0]
    return hidden, w1, w2, topk_w, topk_ids, w1_s, w2_s


def compare(name, cand, ref, atol=0.01, rtol=0.05):
    import torch

    c = cand.to(torch.float32)
    r = ref.to(torch.float32)
    diff = (c - r).abs()
    tol = atol + rtol * r.abs()
    viol = (diff > tol).sum().item()
    max_abs = diff.max().item()
    big = r.abs() > 0.05
    max_rel = (diff[big] / r.abs()[big]).max().item() if big.any() else 0.0
    margin = (tol - diff).min().item()
    ok = viol == 0
    log(
        f"[{name}] {'PASS' if ok else 'FAIL'} viol={viol}/{r.numel()} "
        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} min_margin={margin:.6g} "
        f"(atol={atol}, rtol={rtol})"
    )
    return ok


def run_case(name, model, M, E, H, I, K, seed, sparse_experts=None,
             duplicate_slot=False, noquant=False):
    import torch

    hidden, w1, w2, topk_w, topk_ids, w1_s, w2_s = make_inputs(
        M, E, H, I, K, seed, sparse_experts, duplicate_slot
    )
    cand = model(hidden, w1, w2, topk_w, topk_ids, w1_s, w2_s)
    ref = ref_moe(hidden, w1, w2, topk_w, topk_ids, w1_s, w2_s)
    ok = compare(name, cand, ref)
    if name == "small" and noquant:
        ref_nq = ref_moe(hidden, w1, w2, topk_w, topk_ids, w1_s, w2_s, quant_act=False)
        compare("small-noquant-ref", cand, ref_nq)
    del hidden, w1, w2, topk_w, topk_ids, w1_s, w2_s, cand, ref
    torch.cuda.empty_cache()
    return ok


def parse_args(argv):
    cases = "small,sparse,dup,odd,fourk,sixk"
    noquant = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--cases" and i + 1 < len(argv):
            cases = argv[i + 1]
            i += 2
        elif a == "--noquant":
            noquant = True
            i += 1
        else:
            i += 1
    return cases, noquant


def main():
    log("=== probe env ===")
    log(f"cwd={os.getcwd()}")
    try:
        log(f"listing={sorted(os.listdir('.'))}")
    except Exception as exc:  # noqa: BLE001
        log(f"listing failed: {exc}")
    log(f"python={sys.version.split()[0]}")
    import torch

    log(f"torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"gpu={p.name} cc={p.major}.{p.minor} sms={p.multi_processor_count}")
    import triton

    log(f"triton={triton.__version__}")

    sys.path.insert(0, os.getcwd())
    import kernel

    log(f"kernel_file={getattr(kernel, '__file__', None)}")
    m_bare = kernel.Model()
    log(
        "bare Model() OK: "
        f"num_experts={m_bare.num_experts} intermediate={m_bare.intermediate_size} "
        f"top_k={m_bare.top_k} block_shape={m_bare.block_shape}"
    )

    E, I, K = 256, 512, 8
    H = 2048
    model = kernel.Model(num_experts=E, intermediate_size=I, top_k=K, block_shape=[128, 128])
    model.eval()

    cases_arg, noquant = parse_args(sys.argv[1:])
    cases = cases_arg.split(",")
    results = {}
    specs = {
        "small": lambda: run_case("small", model, 512, E, H, I, K, seed=7, noquant=noquant),
        "sparse": lambda: run_case(
            "sparse", model, 512, E, H, I, K, seed=11, sparse_experts=40
        ),
        "dup": lambda: run_case("dup", model, 512, E, H, I, K, seed=13, duplicate_slot=True),
        "odd": lambda: run_case("odd", model, 457, E, H, I, K, seed=17),
        "fourk": lambda: run_case("fourk", model, 4096, E, H, I, K, seed=19),
        "sixk": lambda: run_case("sixk", model, 6144, E, H, I, K, seed=23),
    }
    for c in cases:
        c = c.strip()
        if not c:
            continue
        try:
            results[c] = specs[c]()
        except Exception as exc:  # noqa: BLE001
            log(f"[{c}] ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results[c] = False

    log("=== summary ===")
    for c, ok in results.items():
        log(f"{c}: {'PASS' if ok else 'FAIL'}")
    log(f"OVERALL={'PASS' if all(results.values()) and results else 'FAIL'}")


if __name__ == "__main__":
    main()
