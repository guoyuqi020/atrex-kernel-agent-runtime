"""Block-scaled FP8 fused MoE implemented directly in CuteDSL (nvidia-cutlass-dsl).

Operator contract (public, from agent_problem.json):
  hidden_states : bfloat16 [token_count, 2048]
  w1            : float8_e4m3fn [256, 1024, 2048]   (gate_up projection weights)
  w2            : float8_e4m3fn [256, 2048, 512]   (down projection weights)
  topk_weights  : float32 [token_count, 8]
  topk_ids      : int32   [token_count, 8]
  w1_scale      : float32 [256, 8, 16]   (128x128 weight-block scales)
  w2_scale      : float32 [256, 16, 4]
  output        : bfloat16 [token_count, 2048]

Pipeline (all GPU compute is performed by self-authored CuteDSL kernels,
launched sequentially on the current stream by one @cute.jit host function):

  1. routing_kernel   : histogram + prefix scan of the flat top-k ids; builds
                        the expert-sorted pair order (padded to GEMM M-tiles)
                        and the per-M-tile expert id table.
  2. quant_kernel     : per-128-group activation quantization of the hidden
                        states (bfloat16 -> e4m3 + float32 group scales).
  3. gemm_kernel (x1) : block-scaled SIMT GEMM, C1 = A_q @ W1_e^T with the
                        128x128 weight-block / 128-K activation scales applied
                        every 128 K elements; writes bfloat16 [pairs, 1024].
  4. silu_quant_kernel: SiLU(gate)*up in the reference's rounding order, then
                        per-128-group requantization of the intermediate rows.
  5. gemm_kernel (x2) : block-scaled SIMT GEMM for the down projection,
                        writes bfloat16 [pairs, 2048].
  6. combine_kernel   : top-k weighted reduction back to [tokens, 2048].

torch is used only for allocating/re-shaping the input/output tensors and for
reading shape metadata; it performs no computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr

# ---------------------------------------------------------------------------
# Operator-contract constants
# ---------------------------------------------------------------------------
HIDDEN = 2048
INTER = 512
N1 = 2 * INTER          # gate_up output width
K1 = HIDDEN             # gate_up K extent
N2 = HIDDEN             # down-projection output width
K2 = INTER              # down-projection K extent
NUM_EXPERTS = 256
TOPK = 8
GROUP = 128             # quantization group size (block_shape [128, 128])
FP8_MAX = 448.0

# ---------------------------------------------------------------------------
# SIMT GEMM tile constants (shared by both GEMM stages)
# ---------------------------------------------------------------------------
BM = 64
BN = 64
BK = 64
GEMM_THREADS = 128
TM = 8                  # micro-tile rows per thread
TN = 4                  # micro-tile cols per thread
ROUT_THREADS = 128
ELEM_THREADS = 256

_gmem = cutlass.AddressSpace.gmem


# ---------------------------------------------------------------------------
# Device kernels (self-authored CuteDSL)
# ---------------------------------------------------------------------------
@cute.kernel
def routing_kernel(
    topk_ids: cute.Tensor,      # int32 [P]
    sorted_ids: cute.Tensor,    # int32 [pad_cap], output
    block_expert: cute.Tensor,  # int32 [grid_m_cap], output
    P: cutlass.Int32,           # number of token-expert pairs
    grid_m_cap: cutlass.Int32,  # capacity of block_expert
):
    tidx, _, _ = cute.arch.thread_idx()

    sm_ptr = cute.arch.alloc_smem(cutlass.Int32, 3 * NUM_EXPERTS, alignment=16)
    sm = cute.make_tensor(
        sm_ptr, cute.make_layout((3, NUM_EXPERTS), stride=(NUM_EXPERTS, 1))
    )

    for i in cutlass.range(tidx, NUM_EXPERTS, ROUT_THREADS):
        sm[0, i] = 0
        sm[1, i] = 0
        sm[2, i] = 0
    cute.arch.barrier()

    if tidx == 0:
        # Histogram of expert ids over all token-expert pairs.
        for p in cutlass.range(P):
            e = topk_ids[p]
            sm[0, e] = sm[0, e] + 1

        # Exclusive prefix sum over per-expert padded token counts plus the
        # per-M-tile expert table. The running cursor and block counter are
        # loop-carried scalars of the dynamic expert loop.
        cursor = cutlass.Int32(0)
        blk = cutlass.Int32(0)
        for e in cutlass.range(NUM_EXPERTS):
            cnt = sm[0, e]
            sm[1, e] = cursor
            sm[2, e] = cursor
            nblk = (cnt + BM - 1) // BM
            for j in cutlass.range(nblk):
                block_expert[blk + j] = e
            blk = blk + nblk
            cursor = cursor + nblk * BM

        # Place each pair into its expert's region.
        for p in cutlass.range(P):
            e = topk_ids[p]
            pos = sm[2, e]
            sorted_ids[pos] = p
            sm[2, e] = pos + 1

        # Pad every expert region with the -1 sentinel.
        for e in cutlass.range(NUM_EXPERTS):
            cnt = sm[0, e]
            end = sm[1, e] + ((cnt + BM - 1) // BM) * BM
            for j in cutlass.range(sm[2, e], end, 1):
                sorted_ids[j] = -1

        # Mark unused M tiles as invalid.
        for b in cutlass.range(blk, grid_m_cap, 1):
            block_expert[b] = -1


@cute.kernel
def quant_kernel(
    x: cute.Tensor,             # bfloat16 [R * C]
    q: cute.Pointer,            # float8_e4m3fn [R * C], output
    s: cute.Pointer,            # float32 [R * C / GROUP], output
    total_groups: cutlass.Int32,
    groups_per_row: cutlass.Int32,
    C: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    g = bidx * ELEM_THREADS + tidx
    if g < total_groups:
        row = g // groups_per_row
        grp = g % groups_per_row
        base = row * C + grp * GROUP
        amax = cutlass.Float32(0.0)
        for i in cutlass.range(GROUP):
            v = x[base + i].to(cutlass.Float32)
            amax = cute.max(amax, cute.abs(v))
        scale = cute.max(amax, cutlass.Float32(1e-12)) / cutlass.Float32(FP8_MAX)
        s[g] = scale
        for i in cutlass.range(GROUP):
            v = x[base + i].to(cutlass.Float32) / scale
            v = cute.clamp(v, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX))
            q[base + i] = v.to(cutlass.Float8E4M3FN)


@cute.kernel
def gemm_kernel(
    a_q: cute.Pointer,          # float8_e4m3fn activation rows x K
    a_scale: cute.Pointer,      # float32 activation rows x (K / GROUP)
    sorted_ids: cute.Tensor,    # int32 [pad_cap]
    block_expert: cute.Tensor,  # int32 [grid_m_cap]
    w: cute.Pointer,            # float8_e4m3fn [E, N, K]
    w_scale: cute.Pointer,      # float32 [E, N / GROUP, K / GROUP]
    c: cute.Tensor,             # bfloat16 [P, N], output (indexed by pair id)
    N: cutlass.Int32,
    K: cutlass.Int32,
    a_shift: cutlass.Int32,     # pair_id >> a_shift == activation row id
):
    tidx, _, _ = cute.arch.thread_idx()
    mb, nb, _ = cute.arch.block_idx()

    sa_ptr = cute.arch.alloc_smem(cutlass.Float8E4M3FN, BM * BK, alignment=16)
    sb_ptr = cute.arch.alloc_smem(cutlass.Float8E4M3FN, BN * BK, alignment=16)
    sp_ptr = cute.arch.alloc_smem(cutlass.Float32, BM * BN, alignment=16)
    sc_ptr = cute.arch.alloc_smem(cutlass.Float32, BM * BN, alignment=16)
    sA = cute.make_tensor(sa_ptr, cute.make_layout((BM, BK), stride=(BK, 1)))
    sB = cute.make_tensor(sb_ptr, cute.make_layout((BN, BK), stride=(BK, 1)))
    sP = cute.make_tensor(sp_ptr, cute.make_layout((BM, BN), stride=(BN, 1)))
    sC = cute.make_tensor(sc_ptr, cute.make_layout((BM, BN), stride=(BN, 1)))

    e = block_expert[mb]
    if e >= 0:
        m0 = mb * BM
        n0 = nb * BN
        k_tiles = K // BK
        scale_nblk = n0 // GROUP
        row_base = (tidx // 16) * TM
        col_base = (tidx % 16) * TN

        # Zero this thread's owned accumulator elements. The tile is exactly
        # covered by the 128 threads' (TM x TN) micro-tiles.
        for i in cutlass.range_constexpr(TM):
            for j in cutlass.range_constexpr(TN):
                sC[row_base + i, col_base + j] = cutlass.Float32(0.0)
                sP[row_base + i, col_base + j] = cutlass.Float32(0.0)

        for kt in cutlass.range(k_tiles):
            k0 = kt * BK
            # Load the A tile (rows of the M tile; sentinel rows -> zeros).
            arow_local = tidx // 2
            acol0 = (tidx % 2) * 32
            pair = sorted_ids[m0 + arow_local]
            arow = pair >> a_shift
            base_a = arow * K + k0 + acol0
            for i in cutlass.range(32):
                v = cutlass.Float8E4M3FN(0.0)
                if pair >= 0:
                    v = a_q[base_a + i]
                sA[arow_local, acol0 + i] = v
            # Load the B tile (weight rows n0..n0+BN, K slice k0..k0+BK).
            brow = tidx // 2
            bcol0 = (tidx % 2) * 32
            base_b = e * (N * K) + (n0 + brow) * K + k0 + bcol0
            for i in cutlass.range(32):
                sB[brow, bcol0 + i] = w[base_b + i]
            cute.arch.barrier()

            # SIMT outer product over the BK slice. The partial accumulator
            # sP lives in shared memory; each of its elements is owned by
            # exactly one thread, so the read-modify-write chain is local.
            # (No Python-list state or static unrolls inside this dynamic
            # loop; everything is staged tensor indexing.)
            for kk in cutlass.range(BK):
                for i in cutlass.range(TM):
                    ri = row_base + i
                    ai = sA[ri, kk].to(cutlass.Float32)
                    for j in cutlass.range(TN):
                        cj = col_base + j
                        sP[ri, cj] = (
                            sP[ri, cj] + ai * sB[cj, kk].to(cutlass.Float32)
                        )
            cute.arch.barrier()

            # Every two K tiles complete one 128-wide activation/weight scale
            # block; fold the block partial sum into the scaled accumulator.
            if (kt % 2) == 1:
                kg = kt // 2
                sw = w_scale[
                    e * ((N // GROUP) * (K // GROUP))
                    + scale_nblk * (K // GROUP)
                    + kg
                ]
                for i in cutlass.range(TM):
                    ri = row_base + i
                    pair_i = sorted_ids[m0 + ri]
                    sa = cutlass.Float32(1.0)
                    if pair_i >= 0:
                        sa = a_scale[(pair_i >> a_shift) * (K // GROUP) + kg]
                    for j in cutlass.range(TN):
                        cj = col_base + j
                        sC[ri, cj] = sC[ri, cj] + sP[ri, cj] * sa * sw
                        sP[ri, cj] = cutlass.Float32(0.0)
            cute.arch.barrier()

        # Epilogue: scatter the bfloat16 result rows back to pair order.
        for i in cutlass.range_constexpr(TM):
            ri = row_base + i
            pair_i = sorted_ids[m0 + ri]
            if pair_i >= 0:
                for j in cutlass.range_constexpr(TN):
                    c[pair_i * N + n0 + col_base + j] = sC[ri, col_base + j].to(
                        cutlass.BFloat16
                    )


@cute.kernel
def silu_quant_kernel(
    c1: cute.Tensor,            # bfloat16 [P, N1] (gate | up)
    inter_b: cute.Tensor,       # bfloat16 [P, INTER], scratch
    inter_q: cute.Pointer,      # float8_e4m3fn [P, INTER], output
    inter_s: cute.Pointer,      # float32 [P, INTER / GROUP], output
    P: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    g = bidx * ELEM_THREADS + tidx
    if g < P * (INTER // GROUP):
        p = g // (INTER // GROUP)
        grp = g % (INTER // GROUP)
        gbase = p * N1 + grp * GROUP
        ubase = gbase + INTER
        obase = p * INTER + grp * GROUP

        # SiLU(gate) * up with the reference rounding order (bfloat16 after
        # the SiLU and after the product), tracking the group amax over the
        # rounded bfloat16 values.
        amax = cutlass.Float32(0.0)
        for i in cutlass.range(GROUP):
            gf = c1[gbase + i].to(cutlass.Float32)
            uf = c1[ubase + i].to(cutlass.Float32)
            sig = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(cutlass.Float32(0.0) - gf)
            )
            sf = (gf * sig).to(cutlass.BFloat16).to(cutlass.Float32)
            v = (sf * uf).to(cutlass.BFloat16)
            inter_b[obase + i] = v
            amax = cute.max(amax, cute.abs(v.to(cutlass.Float32)))

        scale = cute.max(amax, cutlass.Float32(1e-12)) / cutlass.Float32(FP8_MAX)
        inter_s[g] = scale
        for i in cutlass.range(GROUP):
            v = inter_b[obase + i].to(cutlass.Float32) / scale
            v = cute.clamp(v, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX))
            inter_q[obase + i] = v.to(cutlass.Float8E4M3FN)


@cute.kernel
def combine_kernel(
    o2: cute.Tensor,            # bfloat16 [P, HIDDEN]
    topk_w: cute.Tensor,        # float32 [T, TOPK]
    out: cute.Tensor,           # bfloat16 [T, HIDDEN], output
    total: cutlass.Int32,       # T * HIDDEN
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tid = bidx * ELEM_THREADS + tidx
    if tid < total:
        t = tid // HIDDEN
        h = tid % HIDDEN
        acc = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(TOPK):
            pair = t * TOPK + k
            acc = acc + topk_w[pair] * o2[pair * HIDDEN + h].to(cutlass.Float32)
        out[tid] = acc.to(cutlass.BFloat16)


# ---------------------------------------------------------------------------
# Host-side pipeline (one @cute.jit launcher; compiled once per Model)
# ---------------------------------------------------------------------------
@cute.jit
def moe_pipeline(
    hidden: cute.Tensor,        # bfloat16 [T * HIDDEN]
    topk_w: cute.Tensor,        # float32 [T * TOPK]
    topk_ids: cute.Tensor,      # int32 [T * TOPK]
    w1: cute.Pointer,           # float8_e4m3fn [E, N1, K1]
    w1_scale: cute.Pointer,     # float32 [E, N1/GROUP, K1/GROUP]
    w2: cute.Pointer,           # float8_e4m3fn [E, N2, K2]
    w2_scale: cute.Pointer,     # float32 [E, N2/GROUP, K2/GROUP]
    hidden_q: cute.Pointer,     # float8_e4m3fn [T, HIDDEN]
    hidden_s: cute.Pointer,     # float32 [T, HIDDEN/GROUP]
    sorted_ids: cute.Tensor,    # int32 [pad_cap]
    block_expert: cute.Tensor,  # int32 [grid_m_cap]
    c1: cute.Tensor,            # bfloat16 [P, N1]
    inter_b: cute.Tensor,       # bfloat16 [P, INTER]
    inter_q: cute.Pointer,      # float8_e4m3fn [P, INTER]
    inter_s: cute.Pointer,      # float32 [P, INTER/GROUP]
    o2: cute.Tensor,            # bfloat16 [P, N2]
    out: cute.Tensor,           # bfloat16 [T, HIDDEN]
    T: cutlass.Int32,
):
    P = T * TOPK
    pad_cap = P + NUM_EXPERTS * BM + BM
    grid_m_cap = pad_cap // BM

    routing_kernel(topk_ids, sorted_ids, block_expert, P, grid_m_cap).launch(
        grid=[1, 1, 1], block=[ROUT_THREADS, 1, 1]
    )

    groups_h = T * (HIDDEN // GROUP)
    quant_kernel(hidden, hidden_q, hidden_s, groups_h, HIDDEN // GROUP, HIDDEN).launch(
        grid=[cute.ceil_div(groups_h, ELEM_THREADS), 1, 1],
        block=[ELEM_THREADS, 1, 1],
    )

    gemm_kernel(hidden_q, hidden_s, sorted_ids, block_expert, w1, w1_scale, c1, N1, K1, 3).launch(
        grid=[grid_m_cap, N1 // BN, 1], block=[GEMM_THREADS, 1, 1]
    )

    silu_groups = P * (INTER // GROUP)
    silu_quant_kernel(c1, inter_b, inter_q, inter_s, P).launch(
        grid=[cute.ceil_div(silu_groups, ELEM_THREADS), 1, 1],
        block=[ELEM_THREADS, 1, 1],
    )

    gemm_kernel(inter_q, inter_s, sorted_ids, block_expert, w2, w2_scale, o2, N2, K2, 0).launch(
        grid=[grid_m_cap, N2 // BN, 1], block=[GEMM_THREADS, 1, 1]
    )

    combine_kernel(o2, topk_w, out, T * HIDDEN).launch(
        grid=[cute.ceil_div(T * HIDDEN, ELEM_THREADS), 1, 1],
        block=[ELEM_THREADS, 1, 1],
    )


# ---------------------------------------------------------------------------
# Evaluator-facing model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(
        self,
        num_experts: int = NUM_EXPERTS,
        intermediate_size: int = INTER,
        top_k: int = TOPK,
        block_shape: list[int] | tuple[int, int] = (GROUP, GROUP),
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.block_shape = tuple(int(value) for value in block_shape)
        # JIT-compile the whole pipeline once at construction so every
        # forward call is steady-state; the compile cost is paid inside the
        # instantiation budget, not inside any timed forward call.
        self._executor = None
        self._compile_pipeline(torch.device("cuda"))

    # -- compilation -------------------------------------------------------
    def _compile_pipeline(self, device: torch.device) -> None:
        """JIT-compile the CuteDSL pipeline once with representative fakes."""
        t0 = 8
        p0 = t0 * TOPK
        pad_cap0 = p0 + NUM_EXPERTS * BM + BM
        grid_m_cap0 = pad_cap0 // BM

        def t16(n: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(n, device=device, dtype=dtype)

        f = lambda tensor: from_dlpack(tensor, assumed_align=16)  # noqa: E731

        def fptr(tensor, dtype) -> cutlass.Pointer:
            return make_ptr(dtype, tensor.data_ptr(), _gmem, assumed_align=16)

        hidden = f(t16(t0 * HIDDEN, torch.bfloat16))
        topk_w = f(t16(t0 * TOPK, torch.float32))
        topk_ids = f(t16(t0 * TOPK, torch.int32))
        w1 = fptr(t16(1, torch.float8_e4m3fn), cutlass.Float8E4M3FN)
        w1_scale = fptr(t16(1, torch.float32), cutlass.Float32)
        w2 = fptr(t16(1, torch.float8_e4m3fn), cutlass.Float8E4M3FN)
        w2_scale = fptr(t16(1, torch.float32), cutlass.Float32)
        hidden_q = fptr(t16(1, torch.float8_e4m3fn), cutlass.Float8E4M3FN)
        hidden_s = fptr(t16(1, torch.float32), cutlass.Float32)
        sorted_ids = f(t16(pad_cap0, torch.int32))
        block_expert = f(t16(grid_m_cap0, torch.int32))
        c1 = f(t16(p0 * N1, torch.bfloat16))
        inter_b = f(t16(p0 * INTER, torch.bfloat16))
        inter_q = fptr(t16(1, torch.float8_e4m3fn), cutlass.Float8E4M3FN)
        inter_s = fptr(t16(1, torch.float32), cutlass.Float32)
        o2 = f(t16(p0 * N2, torch.bfloat16))
        out = f(t16(t0 * HIDDEN, torch.bfloat16))

        self._executor = cute.compile(
            moe_pipeline,
            hidden,
            topk_w,
            topk_ids,
            w1,
            w1_scale,
            w2,
            w2_scale,
            hidden_q,
            hidden_s,
            sorted_ids,
            block_expert,
            c1,
            inter_b,
            inter_q,
            inter_s,
            o2,
            out,
            t0,
        )

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        device = hidden_states.device

        if self._executor is None:
            self._compile_pipeline(device)

        pairs = token_count * TOPK
        pad_cap = pairs + NUM_EXPERTS * BM + BM
        grid_m_cap = pad_cap // BM

        hidden_states = hidden_states.contiguous()
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        w1_scale = w1_scale.contiguous()
        w2_scale = w2_scale.contiguous()

        hidden_q = torch.empty(token_count, HIDDEN, device=device, dtype=torch.float8_e4m3fn)
        hidden_s = torch.empty(token_count, HIDDEN // GROUP, device=device, dtype=torch.float32)
        sorted_ids = torch.empty(pad_cap, device=device, dtype=torch.int32)
        block_expert = torch.empty(grid_m_cap, device=device, dtype=torch.int32)
        c1 = torch.empty(pairs, N1, device=device, dtype=torch.bfloat16)
        inter_b = torch.empty(pairs, INTER, device=device, dtype=torch.bfloat16)
        inter_q = torch.empty(pairs, INTER, device=device, dtype=torch.float8_e4m3fn)
        inter_s = torch.empty(pairs, INTER // GROUP, device=device, dtype=torch.float32)
        o2 = torch.empty(pairs, N2, device=device, dtype=torch.bfloat16)
        out = torch.empty(token_count, HIDDEN, device=device, dtype=torch.bfloat16)

        def f(tensor: torch.Tensor):
            return from_dlpack(tensor.view(-1), assumed_align=16)

        def fptr(tensor: torch.Tensor, dtype):
            return make_ptr(dtype, tensor.data_ptr(), _gmem, assumed_align=16)

        self._executor(
            f(hidden_states),
            f(topk_weights),
            f(topk_ids),
            fptr(w1, cutlass.Float8E4M3FN),
            fptr(w1_scale, cutlass.Float32),
            fptr(w2, cutlass.Float8E4M3FN),
            fptr(w2_scale, cutlass.Float32),
            fptr(hidden_q, cutlass.Float8E4M3FN),
            fptr(hidden_s, cutlass.Float32),
            f(sorted_ids),
            f(block_expert),
            f(c1),
            f(inter_b),
            fptr(inter_q, cutlass.Float8E4M3FN),
            fptr(inter_s, cutlass.Float32),
            f(o2),
            f(out),
            token_count,
        )
        return out
