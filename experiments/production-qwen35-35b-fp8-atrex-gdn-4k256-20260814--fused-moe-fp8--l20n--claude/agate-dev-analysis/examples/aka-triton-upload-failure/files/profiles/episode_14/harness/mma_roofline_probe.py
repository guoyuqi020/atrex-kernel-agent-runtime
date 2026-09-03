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
                          chains ``acc_i = mma_v2(a, w_i, acc_i)`` over four
                          distinct register-resident W fragments (distinct
                          operands + distinct seeds defeat CSE/licM).
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
import triton.language as tl
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
BM = tl.constexpr(64)
BK = tl.constexpr(128)
BN = tl.constexpr(64)  # per-CTA N; warp slice is [64, 8] under warps_per_cta [1,8]

FLOP_PER_DOT = 2 * 64 * 128 * 64  # one [64,128] x [128,64] fp8 dot


@gluon.jit
def _load_a_tile(a_smem, a_src_ptr):
    rows_a = gl.arange(0, BM, gl.SliceLayout(1, A_BLOCKED))
    cols_a = gl.arange(0, BK, gl.SliceLayout(0, A_BLOCKED))
    cp.async_copy_global_to_shared(
        a_smem, a_src_ptr + rows_a[:, None] * BK + cols_a[None, :]
    )
    cp.commit_group()


@gluon.jit
def _load_w_tile(w_smem, w_src_ptr):
    # Weight source is [BN, BK] (N rows, K contiguous), exactly like the
    # incumbent's w1[e, n, k] indexing: dim0 of the smem tile is K.
    offs_wn = gl.arange(0, BN, gl.SliceLayout(0, W_BLOCKED))
    offs_wk = gl.arange(0, BK, gl.SliceLayout(1, W_BLOCKED))
    cp.async_copy_global_to_shared(
        w_smem, w_src_ptr + offs_wn[None, :] * BK + offs_wk[:, None]
    )
    cp.commit_group()


@gluon.jit
def k1_mma_peak(out_ptr, a_src_ptr, w_src_ptr, ITERS):
    pid = gl.program_id(0)
    mma: gl.constexpr = MMA
    a_dot: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
    w_dot: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=4)

    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [BM, BK], layout=A_SMEM)
    w0_smem = gl.allocate_shared_memory(gl.float8e4nv, [BK, BN], layout=W_SMEM)
    w1_smem = gl.allocate_shared_memory(gl.float8e4nv, [BK, BN], layout=W_SMEM)
    w2_smem = gl.allocate_shared_memory(gl.float8e4nv, [BK, BN], layout=W_SMEM)
    w3_smem = gl.allocate_shared_memory(gl.float8e4nv, [BK, BN], layout=W_SMEM)
    _load_a_tile(a_smem, a_src_ptr)
    w_stride = BK * BN
    _load_w_tile(w0_smem, w_src_ptr + 0 * w_stride)
    _load_w_tile(w1_smem, w_src_ptr + 1 * w_stride)
    _load_w_tile(w2_smem, w_src_ptr + 2 * w_stride)
    _load_w_tile(w3_smem, w_src_ptr + 3 * w_stride)
    cp.wait_group(0)
    gl.barrier()

    a = a_smem.load(a_dot)
    w0 = w0_smem.load(w_dot)
    w1 = w1_smem.load(w_dot)
    w2 = w2_smem.load(w_dot)
    w3 = w3_smem.load(w_dot)

    # Distinct seeds keep the four feedback chains SSA-distinct.
    acc0 = gl.zeros([BM, BN], gl.float32, layout=mma)
    acc1 = gl.full([BM, BN], 1.0e-4, gl.float32, layout=mma)
    acc2 = gl.full([BM, BN], 2.0e-4, gl.float32, layout=mma)
    acc3 = gl.full([BM, BN], 3.0e-4, gl.float32, layout=mma)
    for _ in range(ITERS):
        acc0 = mma_v2(a, w0, acc0)
        acc1 = mma_v2(a, w1, acc1)
        acc2 = mma_v2(a, w2, acc2)
        acc3 = mma_v2(a, w3, acc3)

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
    _load_a_tile(a_smem, a_src_ptr)
    _load_w_tile(wg_smem, wg_src_ptr)
    _load_w_tile(wu_smem, wu_src_ptr)
    cp.wait_group(0)
    gl.barrier()

    a = a_smem.load(a_dot)
    wg = wg_smem.load(w_dot)
    wu = wu_smem.load(w_dot)

    rows_s = gl.arange(0, BM, ROW_SLICE_MMA)
    sg = gl.load(s_src_ptr + rows_s)
    su = gl.load(s_src_ptr + BM + rows_s)
    bg = gl.full([BM], 1.0e-4, gl.float32, layout=ROW_SLICE_MMA)
    bu = gl.full([BM], 1.0e-4, gl.float32, layout=ROW_SLICE_MMA)

    acc_g = gl.zeros([BM, BN], gl.float32, layout=mma)
    acc_u = gl.full([BM, BN], 1.0e-4, gl.float32, layout=mma)
    for _ in range(ITERS):
        acc_g = mma_v2(a, wg, acc_g)
        acc_u = mma_v2(a, wu, acc_u)
        acc_g = acc_g * sg[:, None] + bg[:, None]
        acc_u = acc_u * su[:, None] + bu[:, None]

    tot = gl.sum(gl.convert_layout(acc_g, OUT_BLOCKED))
    tot += gl.sum(gl.convert_layout(acc_u, OUT_BLOCKED))
    gl.store(out_ptr + pid, tot)


def regs_of(kern):
    try:
        caches = getattr(kern, "device_caches", None)
        if not caches:
            return None
        entry = next(iter(caches.values()))
        cache = entry[0] if isinstance(entry, tuple) else entry
        ck = next(iter(cache.values()))
        return {
            "n_regs": getattr(ck, "n_regs", None),
            "n_spills": getattr(ck, "n_spills", None),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {"error": str(exc)}


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

    def fp8_tile(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )

    a_src = fp8_tile(BM, BK)              # [M, K], K contiguous
    w_src = fp8_tile(4, BN, BK)           # four [N, K] tiles, K contiguous
    wg_src = fp8_tile(BN, BK)
    wu_src = fp8_tile(BN, BK)
    s_src = torch.full((2 * BM,), 0.99995, device="cuda", dtype=torch.float32)
    out = torch.empty(2 * num_sms, device="cuda", dtype=torch.float32)

    ITERS = 512
    REPS = 30
    WARMUP = 6
    clocks_before = _clocks_mhz()

    results = {}
    for name, kern, args, dots_per_iter, grid_x in (
        ("k1_mma_peak", k1_mma_peak, (out, a_src, w_src, ITERS), 4, 1),
        ("k1_mma_peak_2cta", k1_mma_peak, (out, a_src, w_src, ITERS), 4, 2),
        ("k2_mma_scaled", k2_mma_scaled, (out, a_src, wg_src, wu_src, s_src, ITERS), 2, 1),
        ("k2_mma_scaled_2cta", k2_mma_scaled, (out, a_src, wg_src, wu_src, s_src, ITERS), 2, 2),
    ):
        grid = (num_sms * grid_x,)
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
        flops = dots_per_iter * FLOP_PER_DOT * ITERS * num_sms * grid_x
        tflops = flops / (med_us * 1e-6) / 1e12
        results[name] = {
            "median_us": round(med_us, 3),
            "min_us": round(times_us[0], 3),
            "max_us": round(times_us[-1], 3),
            "tflops": round(tflops, 2),
            "flops_per_launch": flops,
            "regs": regs_of(kern),
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
