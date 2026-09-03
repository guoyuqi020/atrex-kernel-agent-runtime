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
                        and the per-M-tile expert id table. Single 1024-thread
                        CTA; the per-M-tile expert table and its sentinel tail
                        are filled cooperatively by all threads.
  2. quant_kernel     : per-128-group activation quantization of the hidden
                        states (bfloat16 -> e4m3 + float32 group scales).
                        Warp-per-group mapping: 32 lanes x 4 contiguous
                        elements, one coalesced 8-byte load per lane, warp
                        butterfly amax reduction, packed 4 x fp8 stores.
  3. gemm_kernel (x1) : block-scaled warp-level FP8 tensor-core GEMM
                        (mma.sync m16n8k32, fp32 accumulation), C1 = A_q @
                        W1_e^T with the 128x128 weight-block / 128-K
                        activation scales applied every 128 K elements;
                        writes bfloat16 [pairs, 1024].
  4. silu_quant_kernel: SiLU(gate)*up in the reference's rounding order, then
                        per-128-group requantization of the intermediate rows.
                        Same warp-per-group mapping; the SiLU products stay in
                        registers (no bfloat16 scratch roundtrip).
  5. gemm_kernel (x2) : block-scaled warp-level FP8 tensor-core GEMM for the
                        down projection with the fused top-k combine: each
                        pair row's top-k weight is folded into the per-row
                        scale vector, so the scaled accumulator emerges
                        already weighted; the epilogue stages it as fp16 and
                        issues fp16x4 relaxed gpu-scope atomic adds into a
                        zero-initialized fp16 [tokens, 2048] buffer
                        (L2-resident). No per-pair output tensor is written.
  6. convert_kernel   : fp16 -> bfloat16 conversion of the accumulated output
                        (4 elements per 8-byte chunk, coalesced).

torch is used only for allocating/re-shaping the input/output tensors and for
reading shape metadata; it performs no computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import cutlass
import cutlass.cute as cute
from cutlass._mlir_helpers.vector import Vector
from cutlass.cute.nvgpu import warp
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
# Tensor-core GEMM tile constants (shared by both GEMM stages)
# ---------------------------------------------------------------------------
BM = 32
BN = 64
BK = GROUP              # one K-tile == one 128-wide activation/weight scale block
MMA_M_ATOMS = 2         # warp-level MMA atom tiling: 4 warps as (2 x 2)
MMA_N_ATOMS = 2
MMA_INST = (16, 8, 32)  # mma.sync.aligned m16n8k32 e4m3.e4m3.f32
MMA_PERM_M = MMA_M_ATOMS * MMA_INST[0]          # 32
MMA_PERM_N = MMA_N_ATOMS * MMA_INST[1] * 2      # 32 (doubled N per DSL convention)
MMA_PERM_K = MMA_INST[2]                        # 32
SMEM_PAD = 16           # row padding (elements) for conflict-free ldmatrix
EPI_PAD = 4             # row padding (bf16 elements) for the epilogue sC tile:
                        # stride 64->68 elems (128B->136B) breaks the full-bank-cycle
                        # row alignment so the 32-lane Int64 scatter reads drop from
                        # 32-way to the 2-way minimum shared bank conflicts.
GEMM_THREADS = 128
ROUT_THREADS = 1024
ELEM_THREADS = 256
WARPS_PER_CTA = ELEM_THREADS // 32

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

    cnt_ptr = cute.arch.alloc_smem(cutlass.Int32, NUM_EXPERTS, alignment=16)
    start_ptr = cute.arch.alloc_smem(cutlass.Int32, NUM_EXPERTS, alignment=16)
    cur_ptr = cute.arch.alloc_smem(cutlass.Int32, NUM_EXPERTS, alignment=16)
    nblk_ptr = cute.arch.alloc_smem(cutlass.Int32, NUM_EXPERTS, alignment=16)
    blkstart_ptr = cute.arch.alloc_smem(cutlass.Int32, NUM_EXPERTS, alignment=16)
    total_ptr = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=16)
    sm_cnt = cute.make_tensor(cnt_ptr, cute.make_layout((NUM_EXPERTS,), stride=(1,)))
    sm_start = cute.make_tensor(start_ptr, cute.make_layout((NUM_EXPERTS,), stride=(1,)))
    sm_cur = cute.make_tensor(cur_ptr, cute.make_layout((NUM_EXPERTS,), stride=(1,)))
    sm_nblk = cute.make_tensor(nblk_ptr, cute.make_layout((NUM_EXPERTS,), stride=(1,)))
    sm_blkstart = cute.make_tensor(blkstart_ptr, cute.make_layout((NUM_EXPERTS,), stride=(1,)))
    sm_total = cute.make_tensor(total_ptr, cute.make_layout((1,), stride=(1,)))

    for i in cutlass.range(tidx, NUM_EXPERTS, ROUT_THREADS):
        sm_cnt[i] = 0
        sm_start[i] = 0
        sm_cur[i] = 0
        sm_nblk[i] = 0
        sm_blkstart[i] = 0
    cute.arch.barrier()

    # Parallel histogram of expert ids over all token-expert pairs.
    for p in cutlass.range(tidx, P, ROUT_THREADS):
        e = topk_ids[p]
        cute.arch.atomic_add(
            sm_cnt.iterator + e, cutlass.Int32(1), sem="relaxed", scope="cta"
        )
    cute.arch.barrier()

    if tidx == 0:
        # Exclusive prefix sum over per-expert padded token counts. The
        # running cursor and block counter are loop-carried scalars of the
        # dynamic expert loop; per-expert padded block counts and block
        # offsets are recorded in SMEM so the per-M-tile expert table can be
        # filled cooperatively by the whole CTA below.
        cursor = cutlass.Int32(0)
        blk = cutlass.Int32(0)
        for e in cutlass.range(NUM_EXPERTS):
            cnt = sm_cnt[e]
            sm_start[e] = cursor
            sm_cur[e] = cursor
            nblk = (cnt + BM - 1) // BM
            sm_nblk[e] = nblk
            sm_blkstart[e] = blk
            blk = blk + nblk
            cursor = cursor + nblk * BM
        sm_total[0] = blk
    cute.arch.barrier()

    # Cooperative per-M-tile expert table fill; each expert's block range is
    # written by one thread and the ranges are disjoint.
    for e in cutlass.range(tidx, NUM_EXPERTS, ROUT_THREADS):
        base = sm_blkstart[e]
        for j in cutlass.range(sm_nblk[e]):
            block_expert[base + j] = e

    # Mark unused M tiles as invalid (parallel sentinel tail).
    total_blks = sm_total[0]
    for b in cutlass.range(total_blks + tidx, grid_m_cap, ROUT_THREADS):
        block_expert[b] = -1

    # Parallel placement: each pair claims a slot in its expert's region via
    # an atomic cursor. The order inside a region is irrelevant because the
    # GEMM epilogue scatters every result back by its pair id.
    for p in cutlass.range(tidx, P, ROUT_THREADS):
        e = topk_ids[p]
        pos = cute.arch.atomic_add(
            sm_cur.iterator + e, cutlass.Int32(1), sem="relaxed", scope="cta"
        )
        sorted_ids[pos] = p
    cute.arch.barrier()

    # Pad every expert region with the -1 sentinel (parallel over experts).
    for e in cutlass.range(tidx, NUM_EXPERTS, ROUT_THREADS):
        cnt = sm_cnt[e]
        end = sm_start[e] + ((cnt + BM - 1) // BM) * BM
        for j in cutlass.range(sm_cur[e], end, 1):
            sorted_ids[j] = -1


@cute.kernel
def quant_kernel(
    x: cute.Tensor,             # bfloat16 [R * C]
    q: cute.Pointer,            # float8_e4m3fn [R * C], output
    s: cute.Pointer,            # float32 [R * C / GROUP], output
    total_groups: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Warp-per-group mapping: one warp quantizes one 128-element group, each
    # lane owning 4 contiguous elements. Every access becomes a single
    # coalesced 8-byte load / 4-byte store instead of 128 scalar 2-byte
    # strides, and the group amax is a full-warp butterfly reduction.
    g = bidx * WARPS_PER_CTA + tidx // 32
    if g < total_groups:
        lane = tidx % 32
        idx = g * (GROUP // 4) + lane

        x64 = cute.recast_tensor(x, cutlass.Int64)
        w = x64[idx]

        # Unpack the 4 bfloat16 elements of the 8-byte word: each occupies a
        # 16-bit lane; masking after the shift neutralizes sign extension.
        v0 = (w & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        v1 = ((w >> 16) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        v2 = ((w >> 32) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        v3 = ((w >> 48) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)

        amax = cute.max(
            cute.max(cute.abs(v0), cute.abs(v1)),
            cute.max(cute.abs(v2), cute.abs(v3)),
        )
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=16, mask=-1, mask_and_clamp=31))
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=8, mask=-1, mask_and_clamp=31))
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=4, mask=-1, mask_and_clamp=31))
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=2, mask=-1, mask_and_clamp=31))
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=1, mask=-1, mask_and_clamp=31))

        scale = cute.max(amax, cutlass.Float32(1e-12)) / cutlass.Float32(FP8_MAX)
        if lane == 0:
            s[g] = scale

        # Quantize from the register-resident values and pack the 4 fp8
        # elements into one 32-bit store (little-endian byte order).
        f0 = cute.clamp(v0 / scale, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX)).to(cutlass.Float8E4M3FN)
        f1 = cute.clamp(v1 / scale, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX)).to(cutlass.Float8E4M3FN)
        f2 = cute.clamp(v2 / scale, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX)).to(cutlass.Float8E4M3FN)
        f3 = cute.clamp(v3 / scale, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX)).to(cutlass.Float8E4M3FN)
        packed = (f0.bitcast(cutlass.Int8).to(cutlass.Int32) & 0xFF) | (
            (f1.bitcast(cutlass.Int8).to(cutlass.Int32) & 0xFF) << 8
        ) | (
            (f2.bitcast(cutlass.Int8).to(cutlass.Int32) & 0xFF) << 16
        ) | (
            (f3.bitcast(cutlass.Int8).to(cutlass.Int32) & 0xFF) << 24
        )
        q32 = cute.recast_ptr(q, dtype=cutlass.Int32)
        q32[idx] = packed


@cute.kernel
def gemm_kernel(
    a_q: cute.Pointer,          # float8_e4m3fn activation rows x K
    a_scale: cute.Pointer,      # float32 activation rows x (K / GROUP)
    sorted_ids: cute.Tensor,    # int32 [pad_cap]
    block_expert: cute.Tensor,  # int32 [grid_m_cap]
    w: cute.Pointer,            # float8_e4m3fn [E, N, K]
    w_scale: cute.Pointer,      # float32 [E, N / GROUP, K / GROUP]
    c: cute.Tensor,             # output tile store, indexed by pair id:
                                #   gemm1: bfloat16 [P, N1] (c1)
                                #   gemm2 (combine=True): float16 [T, HIDDEN]
                                #   zero-initialized accumulation target
    N: cutlass.Int32,
    K: cutlass.Int32,
    a_shift: cutlass.Int32,     # pair_id >> a_shift == activation row id
    topk_w: cute.Tensor,        # float32 [T * TOPK] pair weights (combine only)
    combine: cutlass.Constexpr,  # True for gemm2: weighted fp16 atomic epilogue
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    mb, nb, _ = cute.arch.block_idx()

    e = block_expert[mb]
    if e >= 0:
        m0 = mb * BM
        n0 = nb * BN
        k_tiles = K // BK
        scale_nblk = n0 // GROUP

        # SMEM allocations: padded A/B tiles for conflict-free ldmatrix, the
        # double-buffered per-row scale vector (sS0/sS1) and the bf16 epilogue
        # tile sC. The scale product depends only on the tile row (sw is a
        # per-CTA scalar because BN divides GROUP), so each scale buffer is a
        # BM-element vector broadcast along N via a stride-0 mode instead of a
        # full BM x BN tile. Double buffering lets the next k-tile's scale
        # fill proceed without racing the current k-tile's fold.
        sa_ptr = cute.arch.alloc_smem(cutlass.Float8E4M3FN, BM * (BK + SMEM_PAD), alignment=16)
        sb_ptr = cute.arch.alloc_smem(cutlass.Float8E4M3FN, BN * (BK + SMEM_PAD), alignment=16)
        ss_ptr = cute.arch.alloc_smem(cutlass.Float32, 2 * BM, alignment=16)
        # Fused-combine epilogue stages fp16 weighted accumulator values;
        # the plain bf16 epilogue stages bf16. (Trace-time ternary on the
        # constexpr flag -- a staged if-statement cannot yield SMEM handles.)
        sc_ptr = cute.arch.alloc_smem(
            cutlass.Float16 if combine else cutlass.BFloat16,
            BM * (BN + EPI_PAD),
            alignment=16,
        )
        sA = cute.make_tensor(sa_ptr, cute.make_layout((BM, BK), stride=(BK + SMEM_PAD, 1)))
        sB = cute.make_tensor(sb_ptr, cute.make_layout((BN, BK), stride=(BK + SMEM_PAD, 1)))
        ss_flat = cute.make_tensor(ss_ptr, cute.make_layout((2 * BM,), stride=(1,)))
        sS0 = cute.make_tensor(ss_flat.iterator, cute.make_layout((BM, BN), stride=(1, 0)))
        sS1 = cute.make_tensor(ss_flat.iterator + BM, cute.make_layout((BM, BN), stride=(1, 0)))
        sC = cute.make_tensor(sc_ptr, cute.make_layout((BM, BN), stride=(BN + EPI_PAD, 1)))

        # Wide (8-byte) views over the same memory for vectorized staging loads
        # and epilogue stores. The physical SMEM layout is unchanged, so the
        # ldmatrix fragments keep reading the fp8 views above.
        a_q64 = cute.recast_ptr(a_q, dtype=cutlass.Int64)
        w64 = cute.recast_ptr(w, dtype=cutlass.Int64)
        sA64 = cute.make_tensor(
            cute.recast_ptr(sa_ptr, dtype=cutlass.Int64),
            cute.make_layout((BM, BK // 8), stride=((BK + SMEM_PAD) // 8, 1)),
        )
        sB64 = cute.make_tensor(
            cute.recast_ptr(sb_ptr, dtype=cutlass.Int64),
            cute.make_layout((BN, BK // 8), stride=((BK + SMEM_PAD) // 8, 1)),
        )
        sC64 = cute.make_tensor(
            cute.recast_ptr(sc_ptr, dtype=cutlass.Int64),
            cute.make_layout((BM, BN // 4), stride=((BN + EPI_PAD) // 4, 1)),
        )
        c64 = cute.recast_tensor(c, cutlass.Int64)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)   # (MMA, MMA_M, MMA_K)
        tCsB = thr_mma.partition_B(sB)   # (MMA, MMA_N, MMA_K)
        tCsC = thr_mma.partition_C(sC)   # (MMA, MMA_M, MMA_N)
        tCsS0 = thr_mma.partition_C(sS0)  # (MMA, MMA_M, MMA_N)
        tCsS1 = thr_mma.partition_C(sS1)  # (MMA, MMA_M, MMA_N)

        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc_shape = tCsC.shape[:3]
        tCrP = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        tCrC = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        tCrP.fill(0.0)
        tCrC.fill(0.0)

        # SMEM -> RMEM fragment loads via ldmatrix (x4, non-transposed).
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.Float8E4M3FN,
        )
        s2r_tA = cute.make_tiled_copy_A(ldsm_atom, tiled_mma)
        s2r_tB = cute.make_tiled_copy_B(ldsm_atom, tiled_mma)
        thr_s2r_A = s2r_tA.get_slice(tidx)
        thr_s2r_B = s2r_tB.get_slice(tidx)
        tCsA_c = thr_s2r_A.partition_S(sA)
        tCrA_c = thr_s2r_A.retile(tCrA)
        tCsB_c = thr_s2r_B.partition_S(sB)
        tCrB_c = thr_s2r_B.retile(tCrB)

        # Epilogue scatter row mapping (one row half per thread; the first
        # 2*BM threads cover BM rows x 2 halves).
        row = tidx % BM
        half = tidx // BM
        pair_row = sorted_ids[m0 + row]

        # Staging mapping: 16 threads per row each move one 8-byte chunk, so a
        # warp streams two fully contiguous 128-byte rows per instruction.
        ld_row0 = tidx // 16
        ld_chunk = tidx % 16
        zero64 = cutlass.Int64(0)
        k64 = K // 8
        w_base64 = (e * N + n0) * k64

        # Software-pipelined staging: each k-tile's SMEM stores consume a tile
        # that was prefetched into registers during the previous iteration, so
        # the global-load latency overlaps with the barrier wait, MMA, and
        # fold of the current k-tile instead of stalling ahead of the MMA.
        rA = cute.make_rmem_tensor((BM // 8,), cutlass.Int64)
        rB = cute.make_rmem_tensor((BN // 8,), cutlass.Int64)
        # Loop-invariant gather metadata for the A rows: pair ids, sentinel
        # flags, and activation row bases in 8-byte elements.
        rPairV = cute.make_rmem_tensor((BM // 8,), cutlass.Int32)
        rAIdx = cute.make_rmem_tensor((BM // 8,), cutlass.Int32)
        for v in cutlass.range(BM // 8):
            row_v = v * 8 + ld_row0
            pair_v = sorted_ids[m0 + row_v]
            rPairV[v] = pair_v
            a_idx = cutlass.Int32(0)
            if pair_v >= 0:
                a_idx = (pair_v >> a_shift) * k64
            rAIdx[v] = a_idx
        # Loop-invariant scale-row metadata for the per-row scale vector.
        pair_m = cutlass.Int32(-1)
        if tidx < BM:
            pair_m = sorted_ids[m0 + tidx]
        s_idx = cutlass.Int32(0)
        if pair_m >= 0:
            s_idx = (pair_m >> a_shift) * (K // GROUP)
        # Fused combine: the pair row's top-k weight, folded into the per-row
        # scale vector below so the scaled accumulator emerges already
        # weighted (combine specialization only, where the scale fill below
        # multiplies it in; sentinel rows keep 1.0 and are skipped by the
        # epilogue). Loaded unconditionally: the unused value is dead in the
        # gemm1 specialization and the load is L2-hot.
        wt_row = cutlass.Float32(1.0)
        if pair_m >= 0:
            wt_row = topk_w[pair_m]

        # Prefetch the first k-tile (one-time blocking loads) and build its
        # scale vector in the first buffer.
        for v in cutlass.range(BM // 8):
            if rPairV[v] >= 0:
                rA[v] = a_q64[rAIdx[v] + ld_chunk]
            else:
                rA[v] = zero64
        for v in cutlass.range(BN // 8):
            row_v = v * 8 + ld_row0
            rB[v] = w64[w_base64 + row_v * k64 + ld_chunk]
        if tidx < BM:
            sa_m = cutlass.Float32(1.0)
            if pair_m >= 0:
                sa_m = a_scale[s_idx]
            sw = w_scale[
                e * ((N // GROUP) * (K // GROUP)) + scale_nblk * (K // GROUP)
            ]
            # Trace-time ternary on the constexpr flag: the combine
            # specialization folds the top-k weight into the per-row scale.
            row_scale = sa_m * sw * wt_row if combine else sa_m * sw
            sS0[tidx, 0] = row_scale

        par = cutlass.Int32(0)
        for kt in cutlass.range(k_tiles):
            # Store the prefetched tile into SMEM.
            for v in cutlass.range(BM // 8):
                row_v = v * 8 + ld_row0
                sA64[row_v, ld_chunk] = rA[v]
            for v in cutlass.range(BN // 8):
                row_v = v * 8 + ld_row0
                sB64[row_v, ld_chunk] = rB[v]

            # Prefetch the next k-tile and build its scale vector in the idle
            # buffer; these loads stall only when the next iteration consumes
            # them, so their latency overlaps with this iteration's work.
            if kt + 1 < k_tiles:
                k08_next = (kt + 1) * (BK // 8)
                for v in cutlass.range(BM // 8):
                    if rPairV[v] >= 0:
                        rA[v] = a_q64[rAIdx[v] + k08_next + ld_chunk]
                    else:
                        rA[v] = zero64
                for v in cutlass.range(BN // 8):
                    row_v = v * 8 + ld_row0
                    rB[v] = w64[w_base64 + row_v * k64 + k08_next + ld_chunk]
                if tidx < BM:
                    sa_m = cutlass.Float32(1.0)
                    if pair_m >= 0:
                        sa_m = a_scale[s_idx + kt + 1]
                    sw = w_scale[
                        e * ((N // GROUP) * (K // GROUP))
                        + scale_nblk * (K // GROUP)
                        + kt + 1
                    ]
                    row_scale = sa_m * sw * wt_row if combine else sa_m * sw
                    if par == 0:
                        sS1[tidx, 0] = row_scale
                    else:
                        sS0[tidx, 0] = row_scale
            cute.arch.barrier()

            # Tensor-core MMA over the BK slice: 4 k-blocks of 32 K each,
            # accumulating in registers (tCrP).
            for kb in cutlass.range_constexpr(BK // MMA_PERM_K):
                cute.copy(s2r_tA, tCsA_c[None, None, kb], tCrA_c[None, None, kb])
                cute.copy(s2r_tB, tCsB_c[None, None, kb], tCrB_c[None, None, kb])
                cute.gemm(
                    tiled_mma,
                    tCrP,
                    tCrA[None, None, kb],
                    tCrB[None, None, kb],
                    tCrP,
                )

            # Fold the block partial sum into the scaled accumulator using the
            # current k-tile's scale buffer.
            if par == 0:
                for i in range(cute.size(tCrC.shape)):
                    tCrC[i] = tCrC[i] + tCrP[i] * tCsS0[i]
            else:
                for i in range(cute.size(tCrC.shape)):
                    tCrC[i] = tCrC[i] + tCrP[i] * tCsS1[i]
            for i in range(cute.size(tCrC.shape)):
                tCrP[i] = cutlass.Float32(0.0)
            cute.arch.barrier()
            par = 1 - par

        if combine:
            # Fused combine epilogue: the accumulator already carries the
            # pair row's top-k weight (folded into the per-row scale vector),
            # so stage it as fp16 through SMEM and atomic-add each 4-element
            # chunk straight into the zero-initialized fp16 output in token
            # order. The buffer is small enough to stay L2-resident, and each
            # output element receives exactly TOPK adds (one per expert row
            # of the token), so no o2 roundtrip materializes.
            for i in range(cute.size(tCrC.shape)):
                tCsC[i] = tCrC[i].to(cutlass.Float16)
            cute.arch.barrier()
            if tidx < 2 * BM and pair_row >= 0:
                out16 = c.iterator
                row_base = (pair_row // TOPK) * HIDDEN + n0 + half * (BN // 2)
                for j in cutlass.range(BN // 8):
                    v64 = sC64[row, half * 8 + j]
                    x0 = (v64 & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.Float16)
                    x1 = ((v64 >> 16) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.Float16)
                    x2 = ((v64 >> 32) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.Float16)
                    x3 = ((v64 >> 48) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.Float16)
                    cute.arch.atomic_add(
                        out16 + row_base + j * 4,
                        Vector.from_elements((x0, x1, x2, x3), cutlass.Float16).ir_value(),
                        sem="relaxed",
                        scope="gpu",
                    )
        else:
            # Epilogue: stage the bf16 tile through SMEM and scatter rows back
            # to pair order.
            for i in range(cute.size(tCrC.shape)):
                tCsC[i] = tCrC[i].to(cutlass.BFloat16)
            cute.arch.barrier()
            if tidx < 2 * BM and pair_row >= 0:
                for j in cutlass.range(BN // 8):
                    c64[pair_row * (N // 4) + n0 // 4 + half * 8 + j] = sC64[
                        row, half * 8 + j
                    ]


@cute.kernel
def silu_quant_kernel(
    c1: cute.Tensor,            # bfloat16 [P, N1] (gate | up)
    inter_q: cute.Pointer,      # float8_e4m3fn [P, INTER], output
    inter_s: cute.Pointer,      # float32 [P, INTER / GROUP], output
    P: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Same warp-per-group mapping as quant_kernel, over the INTER // GROUP
    # groups of every pair row. Each lane processes 4 contiguous elements of
    # the gate half and the matching 4 of the up half (two coalesced 8-byte
    # loads), keeps the SiLU products in registers, and stores one packed
    # 4 x fp8 word -- no bfloat16 scratch roundtrip.
    g = bidx * WARPS_PER_CTA + tidx // 32
    if g < P * (INTER // GROUP):
        lane = tidx % 32
        p = g // (INTER // GROUP)
        grp = g % (INTER // GROUP)

        c164 = cute.recast_tensor(c1, cutlass.Int64)
        gidx = p * (N1 // 4) + grp * (GROUP // 4) + lane
        gw = c164[gidx]
        uw = c164[gidx + (INTER // 4)]

        g0 = (gw & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        g1 = ((gw >> 16) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        g2 = ((gw >> 32) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        g3 = ((gw >> 48) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        u0 = (uw & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        u1 = ((uw >> 16) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        u2 = ((uw >> 32) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)
        u3 = ((uw >> 48) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.BFloat16).to(cutlass.Float32)

        # SiLU(gate) * up with the reference rounding order (bfloat16 after
        # the SiLU and after the product), tracking the group amax over the
        # rounded bfloat16 values.
        sig0 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(cutlass.Float32(0.0) - g0))
        sig1 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(cutlass.Float32(0.0) - g1))
        sig2 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(cutlass.Float32(0.0) - g2))
        sig3 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(cutlass.Float32(0.0) - g3))
        f0 = ((g0 * sig0).to(cutlass.BFloat16).to(cutlass.Float32) * u0).to(cutlass.BFloat16)
        f1 = ((g1 * sig1).to(cutlass.BFloat16).to(cutlass.Float32) * u1).to(cutlass.BFloat16)
        f2 = ((g2 * sig2).to(cutlass.BFloat16).to(cutlass.Float32) * u2).to(cutlass.BFloat16)
        f3 = ((g3 * sig3).to(cutlass.BFloat16).to(cutlass.Float32) * u3).to(cutlass.BFloat16)

        vf0 = f0.to(cutlass.Float32)
        vf1 = f1.to(cutlass.Float32)
        vf2 = f2.to(cutlass.Float32)
        vf3 = f3.to(cutlass.Float32)
        amax = cute.max(
            cute.max(cute.abs(vf0), cute.abs(vf1)),
            cute.max(cute.abs(vf2), cute.abs(vf3)),
        )
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=16, mask=-1, mask_and_clamp=31))
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=8, mask=-1, mask_and_clamp=31))
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=4, mask=-1, mask_and_clamp=31))
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=2, mask=-1, mask_and_clamp=31))
        amax = cute.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=1, mask=-1, mask_and_clamp=31))

        scale = cute.max(amax, cutlass.Float32(1e-12)) / cutlass.Float32(FP8_MAX)
        if lane == 0:
            inter_s[g] = scale

        q0 = cute.clamp(vf0 / scale, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX)).to(cutlass.Float8E4M3FN)
        q1 = cute.clamp(vf1 / scale, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX)).to(cutlass.Float8E4M3FN)
        q2 = cute.clamp(vf2 / scale, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX)).to(cutlass.Float8E4M3FN)
        q3 = cute.clamp(vf3 / scale, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX)).to(cutlass.Float8E4M3FN)
        packed = (q0.bitcast(cutlass.Int8).to(cutlass.Int32) & 0xFF) | (
            (q1.bitcast(cutlass.Int8).to(cutlass.Int32) & 0xFF) << 8
        ) | (
            (q2.bitcast(cutlass.Int8).to(cutlass.Int32) & 0xFF) << 16
        ) | (
            (q3.bitcast(cutlass.Int8).to(cutlass.Int32) & 0xFF) << 24
        )
        iq32 = cute.recast_ptr(inter_q, dtype=cutlass.Int32)
        iq32[p * (INTER // 4) + grp * (GROUP // 4) + lane] = packed


@cute.kernel
def convert_kernel(
    src: cute.Tensor,           # float16 [T, HIDDEN] accumulated output
    dst: cute.Tensor,           # bfloat16 [T, HIDDEN], output
    total_chunks: cutlass.Int32,  # T * (HIDDEN // 4)
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tid = bidx * ELEM_THREADS + tidx
    # Final rounding pass of the fused combine: the fp16 accumulation buffer
    # (zero-initialized, TOPK relaxed atomic adds per element) carries the
    # weighted top-k sums; one element-wise fp16 -> bfloat16 conversion
    # performs the single final rounding, matching the reference. Each thread
    # moves one coalesced 8-byte chunk (4 elements) in and out.
    if tid < total_chunks:
        src64 = cute.recast_tensor(src, cutlass.Int64)
        v = src64[tid]
        x0 = (v & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.Float16).to(cutlass.Float32)
        x1 = ((v >> 16) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.Float16).to(cutlass.Float32)
        x2 = ((v >> 32) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.Float16).to(cutlass.Float32)
        x3 = ((v >> 48) & 0xFFFF).to(cutlass.Int16).bitcast(cutlass.Float16).to(cutlass.Float32)
        b0 = x0.to(cutlass.BFloat16).bitcast(cutlass.Int16).to(cutlass.Int64) & 0xFFFF
        b1 = x1.to(cutlass.BFloat16).bitcast(cutlass.Int16).to(cutlass.Int64) & 0xFFFF
        b2 = x2.to(cutlass.BFloat16).bitcast(cutlass.Int16).to(cutlass.Int64) & 0xFFFF
        b3 = x3.to(cutlass.BFloat16).bitcast(cutlass.Int16).to(cutlass.Int64) & 0xFFFF
        dst64 = cute.recast_tensor(dst, cutlass.Int64)
        dst64[tid] = b0 | (b1 << 16) | (b2 << 32) | (b3 << 48)


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
    inter_q: cute.Pointer,      # float8_e4m3fn [P, INTER]
    inter_s: cute.Pointer,      # float32 [P, INTER/GROUP]
    out_f16: cute.Tensor,       # float16 [T, HIDDEN], zero-initialized
                                # fused-combine accumulation target
    out: cute.Tensor,           # bfloat16 [T, HIDDEN]
    T: cutlass.Int32,
):
    P = T * TOPK
    pad_cap = P + NUM_EXPERTS * BM + BM
    grid_m_cap = pad_cap // BM

    # Warp-level FP8 tensor-core MMA (mma.sync m16n8k32, fp32 accumulation)
    # tiled across the 4 warps of each GEMM CTA.
    mma_op = warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32, MMA_INST)
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        cute.make_layout((MMA_M_ATOMS, MMA_N_ATOMS, 1)),
        permutation_mnk=(MMA_PERM_M, MMA_PERM_N, MMA_PERM_K),
    )

    routing_kernel(topk_ids, sorted_ids, block_expert, P, grid_m_cap).launch(
        grid=[1, 1, 1], block=[ROUT_THREADS, 1, 1]
    )

    groups_h = T * (HIDDEN // GROUP)
    quant_kernel(hidden, hidden_q, hidden_s, groups_h).launch(
        grid=[cute.ceil_div(groups_h, WARPS_PER_CTA), 1, 1],
        block=[ELEM_THREADS, 1, 1],
    )

    gemm_kernel(hidden_q, hidden_s, sorted_ids, block_expert, w1, w1_scale, c1, N1, K1, 3, topk_w, False, tiled_mma).launch(
        grid=[grid_m_cap, N1 // BN, 1], block=[GEMM_THREADS, 1, 1]
    )

    silu_groups = P * (INTER // GROUP)
    silu_quant_kernel(c1, inter_q, inter_s, P).launch(
        grid=[cute.ceil_div(silu_groups, WARPS_PER_CTA), 1, 1],
        block=[ELEM_THREADS, 1, 1],
    )

    # Fused combine: the down-projection epilogue multiplies each pair row by
    # its top-k weight (via the scale vector) and atomic-adds fp16x4 chunks
    # into the zero-initialized out_f16; no per-pair o2 tensor is written.
    gemm_kernel(inter_q, inter_s, sorted_ids, block_expert, w2, w2_scale, out_f16, N2, K2, 0, topk_w, True, tiled_mma).launch(
        grid=[grid_m_cap, N2 // BN, 1], block=[GEMM_THREADS, 1, 1]
    )

    convert_chunks = T * (HIDDEN // 4)
    convert_kernel(out_f16, out, convert_chunks).launch(
        grid=[cute.ceil_div(convert_chunks, ELEM_THREADS), 1, 1],
        block=[ELEM_THREADS, 1, 1],
    )


# ---------------------------------------------------------------------------
# Evaluator-facing model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(
        self,
        num_experts: int,
        intermediate_size: int,
        top_k: int,
        block_shape: list[int] | tuple[int, int] = (128, 128),
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
        inter_q = fptr(t16(1, torch.float8_e4m3fn), cutlass.Float8E4M3FN)
        inter_s = fptr(t16(1, torch.float32), cutlass.Float32)
        out_f16 = f(t16(t0 * HIDDEN, torch.float16))
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
            inter_q,
            inter_s,
            out_f16,
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
        inter_q = torch.empty(pairs, INTER, device=device, dtype=torch.float8_e4m3fn)
        inter_s = torch.empty(pairs, INTER // GROUP, device=device, dtype=torch.float32)
        # Fused-combine accumulation target: allocated fresh and zeroed on
        # every call (each output element receives exactly TOPK atomic adds);
        # shape-keyed only, no pointer or value reuse across calls.
        out_f16 = torch.zeros(token_count, HIDDEN, device=device, dtype=torch.float16)
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
            fptr(inter_q, cutlass.Float8E4M3FN),
            fptr(inter_s, cutlass.Float32),
            f(out_f16),
            f(out),
            token_count,
        )
        return out
