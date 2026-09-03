#!/usr/bin/env python3
"""Episode-14 roofline probe: fp8 mma_v2 ceilings on the L20N pod.

Primary-source fact (reference-projects/triton AccelerateMatmul.cpp):
computeCapability >= 120 and < 130 supports ONLY MMAv2 (mma.sync); tcgen05 /
TMEM are excluded for consumer Blackwell sm_120. So the question is how much
fp8 throughput mma_v2 itself can deliver on this chip, and how much the
incumbent gate_up kernel's non-mma instruction mix costs.

This probe measures, on one CTA per SM with the incumbent's exact accumulator
layout (NVMMADistributedLayout v[2,0], warps_per_cta [1,8], instrShape 16x8):

  K1 ``k1_mma_peak``   - pure tensor-pipe ceiling: 4 independent feedback
                          chains of ``acc = mma_v2(a, w, acc)`` over
                          register-resident fragments (loaded once from smem).
                          Feedback through the mma defeats LICM/CSE.
  K2 ``k2_mma_scaled`` - gate_up-equivalent issue mix: two dots (gate, up)
                          with feedback mma plus one fp32 scale-fma per
                          element per accumulator per iteration, matching the
                          incumbent's 1:1 mma-instruction : fp32-op ratio.

FLOP accounting counts only mma FLOPs. The probe is a self-authored
development harness: it never touches test_kernel.py, memory/, or the
ground-truth files, and it grades nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

# Incumbent gate_up accumulator layout, verbatim.
MMA: gl.constexpr = gl.NVMMADistributedLayout(
    version=[2, 0], warps_per_cta=[1, 8], instr_shape=[16, 8]
)
A_SMEM: gl.constexpr = gl.SwizzledSharedLayout(
    vec=16, per_phase=1, max_phase=8, order=[1, 0]
)
W_SMEM: gl.constexpr = gl.SwizzledSharedLayout(
    vec=16, per_phase=1, max_phase=8, order=[0, 1]
)
# Incumbent blocked layouts for the cp.async source tiles (tile shapes must be
# multiples of each layout's coverage: A covers [32,128], W covers [128,32]).
A_BLOCKED: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [8, 1], [1, 0])
W_BLOCKED: gl.constexpr = gl.BlockedLayout([16, 1], [8, 4], [1, 8], [0, 1])
OUT_BLOCKED: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0])
ROW_SLICE_MMA: gl.constexpr = gl.SliceLayout(1, MMA)

# Tile constants: identical K footprint to the incumbent k-block (BK=128).
BM: gl.constexpr = 64
BK: gl.constexpr = 128
BN: gl.constexpr = 64  # per-CTA N; warp slice is [64, 8] under warps_per_cta [1,8]

FLOP_PER_DOT = 2 * BM * BK * BN  # one [64,128] x [128,64] fp8 dot


@gluon.jit
def k1_mma_peak(out_ptr, a_src_ptr, w_src_ptr, ITERS):
    pid = gl.program_id(0)
    mma: gl.constexpr = MMA
    a_dot: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
    w_dot: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=4)

    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [BM, BK], layout=A_SMEM)
    w_smem = gl.allocate_shared_memory(gl.float8e4nv, [BK, BN], layout=W_SMEM)
    rows_a = gl.arange(0, BM, gl.SliceLayout(1, A_BLOCKED))
    cols_a = gl.arange(0, BK, gl.SliceLayout(0, A_BLOCKED))
    cp.async_copy_global_to_shared(
        a_smem, a_src_ptr + rows_a[:, None] * BK + cols_a[None, :]
    )
    cp.commit_group()
    rows_w = gl.arange(0, BK, gl.SliceLayout(1, W_BLOCKED))
    cols_w = gl.arange(0, BN, gl.SliceLayout(0, W_BLOCKED))
    cp.async_copy_global_to_shared(
        w_smem, w_src_ptr + rows_w[:, None] * BN + cols_w[None, :]
    )
    cp.commit_group()
    cp.wait_group(0)
    gl.barrier()

    a = a_smem.load(a_dot)
    w = w_smem.load(w_dot)

    acc0 = gl.zeros([BM, BN], gl.float32, layout=mma)
    acc1 = gl.zeros([BM, BN], gl.float32, layout=mma)
    acc2 = gl.zeros([BM, BN], gl.float32, layout=mma)
    acc3 = gl.zeros([BM, BN], gl.float32, layout=mma)
    for _ in range(ITERS):
        acc0 = mma_v2(a, w, acc0)
        acc1 = mma_v2(a, w, acc1)
        acc2 = mma_v2(a, w, acc2)
        acc3 = mma_v2(a, w, acc3)

    tot = gl.sum(gl.convert_layout(acc0, OUT_BLOCKED))
    tot += gl.sum(gl.convert_layout(acc1, OUT_BLOCKED))
    tot += gl.sum(gl.convert_layout(acc2, OUT_BLOCKED))
    tot += gl.sum(gl.convert_layout(acc3, OUT_BLOCKED))
    gl.store(out_ptr + pid, tot)


@gluon.jit
def k2_mma_scaled(out_ptr, a_src_ptr, wg_src_ptr, wu_src_ptr, s_src_ptr, ITERS):
    pid = gl.program_id(0)
    mma: gl.constexpr = MMA
    a_dot: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
    w_dot: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=4)

    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [BM, BK], layout=A_SMEM)
    wg_smem = gl.allocate_shared_memory(gl.float8e4nv, [BK, BN], layout=W_SMEM)
    wu_smem = gl.allocate_shared_memory(gl.float8e4nv, [BK, BN], layout=W_SMEM)
    rows_a = gl.arange(0, BM, gl.SliceLayout(1, A_BLOCKED))
    cols_a = gl.arange(0, BK, gl.SliceLayout(0, A_BLOCKED))
    cp.async_copy_global_to_shared(
        a_smem, a_src_ptr + rows_a[:, None] * BK + cols_a[None, :]
    )
    cp.commit_group()
    rows_w = gl.arange(0, BK, gl.SliceLayout(1, W_BLOCKED))
    cols_w = gl.arange(0, BN, gl.SliceLayout(0, W_BLOCKED))
    cp.async_copy_global_to_shared(
        wg_smem, wg_src_ptr + rows_w[:, None] * BN + cols_w[None, :]
    )
    cp.commit_group()
    cp.async_copy_global_to_shared(
        wu_smem, wu_src_ptr + rows_w[:, None] * BN + cols_w[None, :]
    )
    cp.commit_group()
    cp.wait_group(0)
    gl.barrier()

    a = a_smem.load(a_dot)
    wg = wg_smem.load(w_dot)
    wu = wu_smem.load(w_dot)

    rows_s = gl.arange(0, BM, ROW_SLICE_MMA)
    sg = gl.load(s_src_ptr + rows_s)
    su = gl.load(s_src_ptr + BM + rows_s)
    bg = gl.full([BM], 0.0001, gl.float32, layout=ROW_SLICE_MMA)
    bu = gl.full([BM], 0.0001, gl.float32, layout=ROW_SLICE_MMA)

    acc_g = gl.zeros([BM, BN], gl.float32, layout=mma)
    acc_u = gl.zeros([BM, BN], gl.float32, layout=mma)
    for _ in range(ITERS):
        acc_g = mma_v2(a, wg, acc_g)
        acc_u = mma_v2(a, wu, acc_u)
        acc_g = acc_g * sg[:, None] + bg[:, None]
        acc_u = acc_u * su[:, None] + bu[:, None]

    tot = gl.sum(gl.convert_layout(acc_g, OUT_BLOCKED))
    tot += gl.sum(gl.convert_layout(acc_u, OUT_BLOCKED))
    gl.store(out_ptr + pid, tot)


def _clocks_mhz() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unavailable"
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return f"unavailable ({exc})"


def main() -> int:
    torch.manual_seed(0)
    dev = torch.cuda.get_device_properties(0)
    num_sms = dev.multi_processor_count
    cap = torch.cuda.get_device_capability()

    a_src = torch.randn(BM, BK, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    w_src = torch.randn(BK, BN, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    wg_src = torch.randn(BK, BN, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    wu_src = torch.randn(BK, BN, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    s_src = torch.full((2 * BM,), 0.99995, device="cuda", dtype=torch.float32)
    out = torch.empty(num_sms, device="cuda", dtype=torch.float32)

    ITERS = 512
    REPS = 30
    WARMUP = 6
    grid = (num_sms,)
    clocks_before = _clocks_mhz()

    results = {}
    for name, kern, args, dots_per_iter in (
        ("k1_mma_peak", k1_mma_peak, (out, a_src, w_src, ITERS), 4),
        ("k2_mma_scaled", k2_mma_scaled, (out, a_src, wg_src, wu_src, s_src, ITERS), 2),
    ):
        for _ in range(WARMUP):
            kern[grid](*args, num_warps=8)
        torch.cuda.synchronize()
        times_us = []
        for _ in range(REPS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            kern[grid](*args, num_warps=8)
            e.record()
            torch.cuda.synchronize()
            times_us.append(s.elapsed_time(e) * 1000.0)
        times_us.sort()
        med_us = times_us[len(times_us) // 2]
        flops = dots_per_iter * FLOP_PER_DOT * ITERS * num_sms
        tflops = flops / (med_us * 1e-6) / 1e12
        results[name] = {
            "median_us": round(med_us, 3),
            "min_us": round(times_us[0], 3),
            "max_us": round(times_us[-1], 3),
            "tflops": round(tflops, 2),
            "flops_per_launch": flops,
        }

    clocks_after = _clocks_mhz()
    result = {
        "device_name": dev.name,
        "capability": list(cap),
        "num_sms": num_sms,
        "iters": ITERS,
        "reps": REPS,
        "grid": num_sms,
        "flop_per_dot": FLOP_PER_DOT,
        "kernels": results,
        "clocks_before": clocks_before,
        "clocks_after": clocks_after,
        "torch_version": torch.__version__,
        "collected_at_monotonic": time.monotonic(),
    }
    print(f"[episode14_probe] RESULT_JSON={json.dumps(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
