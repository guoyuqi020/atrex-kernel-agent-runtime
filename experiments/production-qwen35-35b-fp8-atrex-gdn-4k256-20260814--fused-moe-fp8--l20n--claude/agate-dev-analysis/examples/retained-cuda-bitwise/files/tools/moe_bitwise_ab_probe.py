"""Bitwise A/B + regcount probe for fused_moe_fp8 candidate ports (dev pod).

Runs INSIDE a gateway-execute dev job beside the candidate kernel.py and a
reference build uploaded as baseline_kernel.py (usually the pre-edit
incumbent source). For each token count it builds ONE set of contract-shaped
inputs, runs both modules' Models three times each (exercising plain-launch,
graph-capture, and graph-replay paths when the module has a graph path), and
requires bitwise-identical outputs between candidate and baseline and across
repeat calls. Then prints register/smem/occupancy attributes for the
candidate's kernels (use after ANY hot-loop addressing change - this lineage
once silently lost residency to register bloat).

Invocation:
{
  "operation": "dev",
  "command": "python3 moe_bitwise_ab_probe.py [token_count ...]",
  "file_paths": ["tools/moe_bitwise_ab_probe.py", "scratch/baseline_kernel.py"],
  "intent": "custom_harness"
}

Outputs: one `AB tokens=... path=... bitwise_equal=... finite=...` line per
token count, one `AB_VERDICT PASS|FAIL` line, and one `REGPROBE ...` line per
kernel. NOT the evaluator: bitwise equality with the baseline only proves the
candidate did not change the baseline's numerics; the baseline itself must
already be trusted.
"""

import sys

import torch

NUM_EXPERTS = 256
HIDDEN = 2048
INTER = 512
TOP_K = 8
GROUP = 128


def make_inputs(t, seed):
    torch.manual_seed(seed)
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
    tw = (tw / tw.sum(dim=1, keepdim=True)).to(torch.float32)
    return hidden, w1, w2, tw, ids, w1s, w2s


def run3(model, args):
    outs = []
    for _ in range(3):
        out = model(*args)
        torch.cuda.synchronize()
        outs.append(out.clone())
    return outs


def main():
    tokens = [int(a) for a in sys.argv[1:]] or [450, 512, 4096]

    import kernel as cand_mod
    import baseline_kernel as base_mod

    cand = cand_mod.Model(num_experts=NUM_EXPERTS, intermediate_size=INTER,
                          top_k=TOP_K, block_shape=[GROUP, GROUP])
    base = base_mod.Model(num_experts=NUM_EXPERTS, intermediate_size=INTER,
                          top_k=TOP_K, block_shape=[GROUP, GROUP])

    ok = True
    for i, t in enumerate(tokens):
        args = make_inputs(t, 100 + i)
        outs_c = run3(cand, args)
        outs_b = run3(base, args)
        for k in range(3):
            ok &= torch.equal(outs_c[k], outs_b[k])
            if k > 0:
                ok &= torch.equal(outs_c[k], outs_c[0])
                ok &= torch.equal(outs_b[k], outs_b[0])
        tile = "t32" if t * TOP_K <= NUM_EXPERTS * 32 else "t64"
        print("AB tokens=%d path=%s bitwise_equal=%s finite=%s" %
              (t, tile, torch.equal(outs_c[2], outs_b[2]),
               torch.isfinite(outs_c[2].float()).all().item()))
    print("AB_VERDICT", "PASS" if ok else "FAIL")

    # Regcount / occupancy probe on the candidate runtime.
    rt = cand_mod._RUNTIME_CACHE[0]
    attr = rt.cu.CUfunction_attribute
    for name, threads in (("k_gemm_t64", 256), ("k_gemm_t32", 128),
                          ("k_silu_quant", 256), ("k_quant_act", 256),
                          ("k_reduce", 256)):
        func = rt.funcs[name]
        _, regs = rt.cu.cuFuncGetAttribute(
            attr.CU_FUNC_ATTRIBUTE_NUM_REGS, func)
        _, smem = rt.cu.cuFuncGetAttribute(
            attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func)
        _, local = rt.cu.cuFuncGetAttribute(
            attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func)
        _, nblk = rt.cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            func, threads, 0)
        print("REGPROBE kernel=%s threads=%d regs=%d static_smem=%d "
              "local_bytes=%d max_blocks_per_sm=%d" %
              (name, threads, regs, smem, local, nblk))


if __name__ == "__main__":
    main()
