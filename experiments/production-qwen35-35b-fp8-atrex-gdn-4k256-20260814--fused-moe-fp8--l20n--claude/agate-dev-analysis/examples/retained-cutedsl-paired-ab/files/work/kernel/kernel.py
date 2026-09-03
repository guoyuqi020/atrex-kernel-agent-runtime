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

  1. routing (4 small kernels): routing_init_kernel sentinel-fills sorted_ids /
                         block_expert and zeros the per-expert bookkeeping;
                         routing_hist_kernel builds the expert histogram with
                         one atomic add per pair; routing_scan_kernel computes
                         the 256 padded per-expert extents in parallel, prefixes
                         them in smem, and writes the per-M-tile expert table
                         plus the per-M-tile valid-row count (tile_limit: BM
                         for full tiles, the expert's remainder for its last
                         tile) with one thread per expert;
                         routing_place_kernel scatters every pair to its
                         expert region with an atomic ticket, recording the
                         inverse map pos_of_pair[pair] = sorted position. The
                         intra-expert order is race-determined, which is
                         harmless: every downstream stage indexes rows through
                         pos_of_pair, and each pair's output is independent.
  2. quant_gather_kernel : fused per-128-group activation quantization and
                         pair gather: two blocks per token each quantize the
                         row (bfloat16 -> e4m3 + float32 group scales,
                         identical values in both blocks) and write half of
                         the token's 8 sorted pair positions from
                         pos_of_pair DIRECTLY in expert-sorted position
                         order, so both GEMMs read dense contiguous A tiles
                         without any token-order fp8 intermediate; sentinel
                         positions are skipped (their rows are never read by
                         any downstream stage).
  3. gemm1_fused_kernel : block-scaled tensor-core (mma.sync.m16n8k32 e4m3)
                         GEMM for the gate_up projection WITH the SiLU(gate)*up
                         activation and the per-128-group intermediate
                         requantization fused into its epilogue. Each CTA owns
                         one (gate, up) N-tile pair: it sweeps K for the gate
                         tile, stages the bf16-rounded gate accumulator (the
                         exact values the old c1 store materialized), sweeps K
                         for the up tile, then evaluates SiLU(gate)*up in the
                         reference's rounding order and stores inter_q /
                         inter_s directly (tile_limit-guarded, valid rows
                         only). This deletes the c1 buffer and its DRAM
                         round-trip plus the old silu_quant kernel entirely.
  4. gemm_tc_kernel     : tensor-core GEMM for the down projection. Its
                         epilogue stores only each tile's tile_limit valid
                         rows: expert BM-rounding padding rows are computed
                         but never read downstream, so their stores are
                         skipped.
  5. combine_kernel    : top-k weighted reduction back to [tokens, 2048].

All intermediate [pairs, *] buffers are stored in expert-sorted position
order (indexed by sorted position, not pair id); the pos_of_pair inverse
map connects them to the per-pair routing weights in the combine stage.

torch is used only for allocating/re-shaping the input/output tensors and for
reading shape metadata; it performs no computation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
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
BM = 64                 # CTA tile rows (M)
BN = 128                # CTA tile cols (N) == one 128-wide weight scale block
BK = 128                # CTA tile K == one 128-K activation/weight scale group
RASTER_GROUP_M = 8      # M tiles swept over all N tiles before advancing (L2 group)
GEMM_STAGES = 2         # double-buffered cp.async smem pipeline
GEMM_THREADS = 128      # 4 warps, atom layout (2, 2, 1)
MMA_INST = (16, 8, 32)  # mma.sync.m16n8k32 e4m3 -> f32
ATOM_LAYOUT = (2, 2, 1)
NUM_K_BLOCK = BK // MMA_INST[2]
# Scale-fold row classes of this tiled MMA's C fragment (probe-verified in pod
# for all 128 threads): each thread's 64 accumulator elements touch exactly 4
# distinct M rows, and the element->class map below is the SAME for every
# thread. Class c is represented by element index c (elements 0, 2, 4, 6), so
# the combined scale for class c is a_scale[m0 + tCcI[c][0], kt] * w_scale.
FOLD_CLASS = (0, 0, 2, 2, 4, 4, 6, 6) * 8
FOLD_CLASS_IDX = tuple(c // 2 for c in FOLD_CLASS)  # 0..3 index into the 4 regs
ELEM_THREADS = 256
QUANT_THREADS = 128           # quant block size: exact grids for any T
SUB_ELEMS = 16                # contiguous elements per thread (128-bit bf16 x2 / fp8 x1)
SUBS_PER_GROUP = GROUP // SUB_ELEMS
F8 = cutlass.Float8E4M3FN

# ---------------------------------------------------------------------------
# gemm1-fused tile constants. The fused gate_up GEMM keeps BK == GROUP so each
# k tile is exactly one scale group (fold every tile). If register pressure of
# the staged gate fragment ever forces it, FUSED_BK = 64 selects the fallback
# branch: half-group k tiles (fold every 2 tiles, bitwise-verified pattern),
# pipeline smem shrinks 48KB -> 24KB, and the gate accumulator is staged in a
# dedicated 16KB smem region instead of registers (40KB total, still 2 CTAs/SM
# on the 100KB/SM device).
# ---------------------------------------------------------------------------
FUSED_BK = 128
GATE_IN_SMEM = FUSED_BK != GROUP
FUSED_NUM_K_BLOCK = FUSED_BK // MMA_INST[2]
FUSED_ACC = BM * BN // GEMM_THREADS        # accumulator elements per thread
FUSED_UNITS = BM * SUBS_PER_GROUP          # epilogue quant units (rows x subs)

_gmem = cutlass.AddressSpace.gmem


# ---------------------------------------------------------------------------
# GEMM construction helpers (traced at compile time inside the @cute.jit host)
# ---------------------------------------------------------------------------
def _make_smem_layout(tile_m, tile_k, stages):
    """K-major swizzled smem layout for 128-bit cp.async / ldmatrix.b8."""
    major = min(tile_k, 128 * 8 // F8.width)
    swizzle_bits = min(int(math.log2(major * F8.width // 128)), 3)
    base_bits = int(math.log2(128 // 8))
    shift_bits = int(math.log2(128 // F8.width))
    swizzle = cute.make_swizzle(swizzle_bits, base_bits, shift_bits)
    atom = cute.make_layout((8, major), stride=(major, 1))
    layout = cute.tile_to_shape(atom, (tile_m, tile_k, stages), order=(0, 1, 2))
    return layout, swizzle


def _make_gmem_tiled_copy(threads, tile_k):
    atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        F8,
        num_bits_per_copy=128,
    )
    copy_elems = 128 // F8.width
    shape_dim_1 = tile_k // copy_elems
    thread_layout = cute.make_layout(
        (threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
    )
    value_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(atom, thread_layout, value_layout)


def _make_tiled_mma_fp8():
    op = cute.nvgpu.warp.MmaFP8Op(F8, cutlass.Float32, MMA_INST)
    return cute.make_tiled_mma(
        op,
        cute.make_layout(ATOM_LAYOUT),
        permutation_mnk=(
            ATOM_LAYOUT[0] * MMA_INST[0],
            ATOM_LAYOUT[1] * MMA_INST[1] * 2,
            MMA_INST[2],
        ),
    )


# ---------------------------------------------------------------------------
# Device kernels (self-authored CuteDSL)
# ---------------------------------------------------------------------------
@cute.kernel
def routing_init_kernel(
    sorted_ids: cute.Tensor,    # int32 [pad_cap], output: sentinel fill
    block_expert: cute.Tensor,  # int32 [grid_m_cap], output: sentinel fill
    rbuf: cute.Pointer,         # int32 [2 * NUM_EXPERTS], output: zeroed
    n_sorted: cutlass.Int32,    # pad_cap
    n_blocks: cutlass.Int32,    # grid_m_cap
):
    # One pass over the concatenation [sorted_ids | block_expert | rbuf].
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    g = bidx * ELEM_THREADS + tidx
    n = n_sorted + n_blocks + 2 * NUM_EXPERTS
    if g < n:
        if g < n_sorted:
            sorted_ids[g] = -1
        elif g < n_sorted + n_blocks:
            block_expert[g - n_sorted] = -1
        else:
            rbuf[g - n_sorted - n_blocks] = 0


@cute.kernel
def routing_hist_kernel(
    topk_ids: cute.Tensor,  # int32 [P]
    rbuf: cute.Pointer,     # int32 [2 * NUM_EXPERTS], counts section
    P: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    g = bidx * ELEM_THREADS + tidx
    if g < P:
        cute.arch.atomic_add(rbuf + topk_ids[g], 1)


@cute.kernel
def routing_scan_kernel(
    rbuf: cute.Pointer,         # int32 [2 * NUM_EXPERTS]: counts -> offsets
    block_expert: cute.Tensor,  # int32 [grid_m_cap], output
    tile_limit: cute.Tensor,    # int32 [grid_m_cap], output: valid rows/tile
):
    # One block of NUM_EXPERTS threads: per-expert padded block counts in
    # parallel, then a single-thread exclusive prefix over those counts held in
    # smem (no gmem store inside the dependent chain), then a fully parallel
    # write phase where each expert thread emits its own position offset and
    # fills its own M-tile range. This emits exactly the block_expert and
    # rbuf[E+e] values the old serial scan produced (index-exact, so all
    # downstream bytes are unchanged) but moves every gmem store out of the
    # 256-step dependent loop that made the old kernel latency-bound.
    # tile_limit[off+j] records how many rows of M tile off+j are real pairs
    # (BM for every full tile, the expert's remainder for its last tile); it
    # is written in the same fill loop as block_expert, so block_expert[mb] >= 0
    # implies tile_limit[mb] was written. The GEMM epilogues use it to skip
    # stores of never-read padding rows.
    tidx, _, _ = cute.arch.thread_idx()

    sm_ptr = cute.arch.alloc_smem(cutlass.Int32, 2 * NUM_EXPERTS, alignment=16)
    sm_nblk = cute.make_tensor(sm_ptr, cute.make_layout(NUM_EXPERTS))
    sm_off = cute.make_tensor(sm_ptr + NUM_EXPERTS, cute.make_layout(NUM_EXPERTS))

    cnt = rbuf[tidx]
    sm_nblk[tidx] = (cnt + BM - 1) // BM
    cute.arch.barrier()

    if tidx == 0:
        cursor = cutlass.Int32(0)
        for e in cutlass.range(NUM_EXPERTS):
            sm_off[e] = cursor
            cursor = cursor + sm_nblk[e]
    cute.arch.barrier()

    off = sm_off[tidx]
    nb = sm_nblk[tidx]
    # Position-unit ticket base for routing_place (blocks * BM), and the
    # per-M-tile expert fill for this expert's (disjoint) range. All tiles are
    # full (limit BM) except the expert's last, whose limit is the remainder.
    rbuf[NUM_EXPERTS + tidx] = off * BM
    if nb > 0:
        for j in cutlass.range(nb - 1):
            block_expert[off + j] = tidx
            tile_limit[off + j] = BM
        block_expert[off + nb - 1] = tidx
        tile_limit[off + nb - 1] = cnt - (nb - 1) * BM


@cute.kernel
def routing_place_kernel(
    topk_ids: cute.Tensor,    # int32 [P]
    sorted_ids: cute.Tensor,  # int32 [pad_cap], output
    pos_of_pair: cute.Tensor, # int32 [P], output (inverse of sorted_ids)
    rbuf: cute.Pointer,       # int32 [2 * NUM_EXPERTS], offsets section
    P: cutlass.Int32,
):
    # Atomic ticket per pair inside its expert's region; the intra-expert
    # order is race-determined but downstream stages index every row through
    # pos_of_pair, so the permutation is numerically invisible.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    g = bidx * ELEM_THREADS + tidx
    if g < P:
        e = topk_ids[g]
        pos = cute.arch.atomic_add(rbuf + (NUM_EXPERTS + e), 1)
        sorted_ids[pos] = g
        pos_of_pair[g] = pos


@cute.kernel
def quant_gather_kernel(
    x: cute.Pointer,            # bfloat16 [T * HIDDEN], token order
    pos_of_pair: cute.Tensor,   # int32 [P], sorted position of each pair
    q: cute.Pointer,            # float8_e4m3fn [pad_cap * HIDDEN], output,
                                # position order
    s: cute.Pointer,            # float32 [pad_cap * HIDDEN/GROUP], output,
                                # position order
):
    # Fused quantization + pair gather. EXACT grid: 2 blocks per token x
    # QUANT_THREADS threads = HIDDEN/GROUP groups x SUBS_PER_GROUP
    # sub-threads, so no predication is needed and every barrier is reached
    # by all threads. The two blocks of a token redundantly compute the same
    # amax/scale/fq (the source row is read twice, hitting L2) but each
    # writes only half of the token's TOPK sorted pair positions, keeping
    # the per-thread scattered-store count low and the store parallelism
    # high (a one-block-per-token variant serialized 8 scattered stores per
    # thread at 1/9 the old two-stage scheme's thread count and lost the
    # official large-T measurement). The amax reduction, scale and rounding
    # chain are exactly the old quant_kernel's; the difference is where the
    # bytes land: each thread's SUB_ELEMS fp8 values (and each group
    # leader's scale) are written straight into sorted pair positions from
    # pos_of_pair, deleting the token-order fp8 buffer and the separate
    # gather pass that used to permute it. Sentinel positions are still
    # never written (pair positions are the exact image of pos_of_pair) and
    # never read downstream: the tensor-core GEMMs early-exit on empty M
    # tiles (block_expert < 0), and padded tail rows inside a used tile
    # produce garbage that combine never reads (it indexes real positions
    # only, through pos_of_pair). Stores are staged through the register
    # fragment fq (fragment->gmem autovec_copy emits 128-bit stores; a
    # direct gmem->gmem copy would emit byte copies), and per store
    # instruction a warp covers 4 contiguous 128-byte segments of one pair
    # row, so coalescing matches the old gather_kernel.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    sm_ptr = cute.arch.alloc_smem(
        cutlass.Float32, QUANT_THREADS + QUANT_THREADS // SUBS_PER_GROUP,
        alignment=16,
    )
    s_part = cute.make_tensor(sm_ptr, cute.make_layout(QUANT_THREADS))
    s_scale = cute.make_tensor(
        sm_ptr + QUANT_THREADS,
        cute.make_layout(QUANT_THREADS // SUBS_PER_GROUP),
    )

    token = bidx >> 1
    half = bidx % 2
    grp = tidx // SUBS_PER_GROUP
    sub = tidx % SUBS_PER_GROUP
    base = token * HIDDEN + grp * GROUP + sub * SUB_ELEMS

    x16 = cute.make_tensor(x + base, cute.make_layout(SUB_ELEMS))
    fr = cute.make_fragment_like(x16, cutlass.BFloat16)
    cute.autovec_copy(x16, fr)
    amax = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(SUB_ELEMS):
        amax = cute.max(amax, cute.abs(fr[i].to(cutlass.Float32)))
    s_part[tidx] = amax
    cute.arch.barrier()

    gid = tidx // SUBS_PER_GROUP
    if tidx % SUBS_PER_GROUP == 0:
        m = s_part[tidx]
        for k in cutlass.range_constexpr(SUBS_PER_GROUP - 1):
            m = cute.max(m, s_part[tidx + 1 + k])
        scale = cute.max(m, cutlass.Float32(1e-12)) / cutlass.Float32(FP8_MAX)
        s_scale[gid] = scale
    cute.arch.barrier()

    scale = s_scale[gid]
    fq = cute.make_fragment_like(cute.make_layout(SUB_ELEMS), F8)
    for i in cutlass.range_constexpr(SUB_ELEMS):
        v = fr[i].to(cutlass.Float32) / scale
        v = cute.clamp(v, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX))
        fq[i] = v.to(F8)

    # Emit the identical 16 fp8 bytes (and the group scale) to this block's
    # half of the token's TOPK sorted pair positions. All indices are
    # computed unguarded; only the per-position scale store is
    # side-effect-guarded by the sub-leader predicate.
    pair0 = token * TOPK + half * (TOPK // 2)
    dst_col = grp * GROUP + sub * SUB_ELEMS
    groups_per_row = HIDDEN // GROUP
    for k in cutlass.range_constexpr(TOPK // 2):
        pos = pos_of_pair[pair0 + k]
        q16 = cute.make_tensor(
            q + pos * HIDDEN + dst_col, cute.make_layout(SUB_ELEMS)
        )
        cute.autovec_copy(fq, q16)
        if sub == 0:
            s[pos * groups_per_row + grp] = scale


# Branch selection is done at MODULE scope (plain Python at import time, never
# traced): each variant below is a fully straight-line kernel body containing
# only sanctioned dynamic ifs (predicates over traced values) and
# range_constexpr unrolled loops. No python-bool control flow appears inside a
# traced kernel, and SharedStorage is defined exactly once per variant.
if not GATE_IN_SMEM:

    @cute.kernel
    def gemm1_fused_kernel(
        mA: cute.Tensor,            # float8_e4m3fn (pad_cap, K1), position order
        a_scale: cute.Tensor,       # float32 (pad_cap, K1 / GROUP), position order
        block_expert: cute.Tensor,  # int32 [grid_m_cap]
        tile_limit: cute.Tensor,    # int32 [grid_m_cap]: valid rows per M tile
        mW: cute.Tensor,            # float8_e4m3fn (E, N1, K1)
        w_scale: cute.Pointer,      # float32 [E, N1/GROUP, K1/GROUP]
        inter_q: cute.Pointer,      # float8_e4m3fn [pad_cap, INTER], output,
                                    # position order
        inter_s: cute.Pointer,      # float32 [pad_cap, INTER/GROUP], output,
                                    # position order
        K: cutlass.Int32,           # K1
        k_groups: cutlass.Int32,    # K1 / GROUP
        n_groups: cutlass.Int32,    # N1 / GROUP (w_scale stride; 2x pair count)
        num_m: cutlass.Int32,       # grid_m_pad (number of M tiles)
        sA_layout: cute.Layout,
        sA_swz: cute.Swizzle,
        sB_layout: cute.Layout,
        sB_swz: cute.Swizzle,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        # Register-staged variant (FUSED_BK == GROUP): one k tile == one scale
        # group, fold every tile. The gate accumulator's bf16-rounded values
        # (EXACTLY the values the old c1 gmem store materialized) are held in
        # a register fragment across the up sweep; the epilogue evaluates the
        # old silu_quant rounding chain on them and stores inter_q / inter_s
        # directly, deleting the c1 round-trip and the silu_quant kernel.
        tidx, _, _ = cute.arch.thread_idx()
        L, _, _ = cute.arch.block_idx()

        # Branch-free L2-grouped rasterization (as gemm_tc_kernel) over the
        # INTER/BN == n_groups/2 pair columns instead of all N tiles.
        num_n = n_groups // 2
        group_tiles = RASTER_GROUP_M * num_n
        group = L // group_tiles
        r = L % group_tiles
        mb = group * RASTER_GROUP_M + r % RASTER_GROUP_M
        nb = r // RASTER_GROUP_M

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[F8, cute.cosize(sA_layout)], 16
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[F8, cute.cosize(sB_layout)], 16
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
        sA = SharedStorage(storage).a.get_tensor(sA_layout, swizzle=sA_swz)
        sB = SharedStorage(storage).b.get_tensor(sB_layout, swizzle=sB_swz)

        e = block_expert[mb]
        if e >= 0:
            m0 = mb * BM
            k_tiles = K // FUSED_BK
            n_blk_g = nb                        # weight block of the gate half
            n_blk_u = nb + INTER // GROUP       # weight block of the up half

            gW = mW[e, None, None]
            gA_mk = cute.local_tile(mA, (BM, FUSED_BK), (mb, None))
            gBg_nk = cute.local_tile(gW, (BN, FUSED_BK), (nb, None))
            gBu_nk = cute.local_tile(gW, (BN, FUSED_BK), (n_blk_u, None))

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA_mk)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gBg_nk)
            tBuB = thr_copy_B.partition_S(gBu_nk)
            tBsB = thr_copy_B.partition_D(sB)

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            # Identity tensor: per-accumulator-element (m, n) CTA-tile coords
            # (fragment shape, scale-fold row lookup, epilogue v placement).
            cI = cute.make_identity_tensor((BM, BN))
            tCcI = thr_mma.partition_C(cI)
            tCrC = tiled_mma.make_fragment_C(tCcI.layout)
            tCrC_raw = cute.make_fragment_like(tCrC)
            tCrC.fill(0.0)

            atom_s2r = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x16x8bOp(False, 4), F8
            )
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r, tiled_mma)
            thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_cv = thr_s2r_A.partition_S(sA)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrA_cv = thr_s2r_A.retile(tCrA)
            tCsB_cv = thr_s2r_B.partition_S(sB)
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrB_cv = thr_s2r_B.retile(tCrB)

            # Register-fold row classes (FOLD_CLASS): elements 0, 2, 4, 6 give
            # the 4 distinct M rows this thread's accumulator touches.
            fold_row = (tCcI[0][0], tCcI[2][0], tCcI[4][0], tCcI[6][0])

            # Gate staging fragment (live across the up sweep).
            tCrC_gate_bf = cute.make_fragment_like(
                cute.make_layout(FUSED_ACC), cutlass.BFloat16
            )

            # ---- gate K sweep -------------------------------------------
            cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
            cute.copy(tiled_copy_B, tBgB[None, None, None, 0], tBsB[None, None, None, 0])
            cute.arch.cp_async_commit_group()

            pipe_read = cutlass.Int32(0)
            for kt in cutlass.range(k_tiles):
                if kt + 1 < k_tiles:
                    wr = (kt + 1) % GEMM_STAGES
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, kt + 1],
                        tAsA[None, None, None, wr],
                    )
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, kt + 1],
                        tBsB[None, None, None, wr],
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                sw = w_scale[e * n_groups * k_groups + n_blk_g * k_groups + kt]
                rs = (
                    a_scale[m0 + fold_row[0], kt] * sw,
                    a_scale[m0 + fold_row[1], kt] * sw,
                    a_scale[m0 + fold_row[2], kt] * sw,
                    a_scale[m0 + fold_row[3], kt] * sw,
                )

                tCrC_raw.fill(0.0)
                for kb in cutlass.range_constexpr(FUSED_NUM_K_BLOCK):
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_cv[None, None, kb, pipe_read],
                        tCrA_cv[None, None, kb],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_cv[None, None, kb, pipe_read],
                        tCrB_cv[None, None, kb],
                    )
                    cute.gemm(
                        tiled_mma,
                        tCrC_raw,
                        tCrA[None, None, kb],
                        tCrB[None, None, kb],
                        tCrC_raw,
                    )

                for i in cutlass.range_constexpr(cute.size(tCrC)):
                    tCrC[i] = tCrC[i] + tCrC_raw[i] * rs[FOLD_CLASS_IDX[i]]

                cute.arch.sync_threads()
                pipe_read = (kt + 1) % GEMM_STAGES

            # Stage the gate accumulator's bf16-rounded values (the exact
            # bytes the old c1 store materialized) for the fused epilogue.
            for i in cutlass.range_constexpr(FUSED_ACC):
                tCrC_gate_bf[i] = tCrC[i].to(cutlass.BFloat16)

            # ---- up K sweep ------------------------------------------------
            tCrC.fill(0.0)
            cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
            cute.copy(tiled_copy_B, tBuB[None, None, None, 0], tBsB[None, None, None, 0])
            cute.arch.cp_async_commit_group()

            pipe_read = cutlass.Int32(0)
            for kt in cutlass.range(k_tiles):
                if kt + 1 < k_tiles:
                    wr = (kt + 1) % GEMM_STAGES
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, kt + 1],
                        tAsA[None, None, None, wr],
                    )
                    cute.copy(
                        tiled_copy_B,
                        tBuB[None, None, None, kt + 1],
                        tBsB[None, None, None, wr],
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                sw = w_scale[e * n_groups * k_groups + n_blk_u * k_groups + kt]
                rs = (
                    a_scale[m0 + fold_row[0], kt] * sw,
                    a_scale[m0 + fold_row[1], kt] * sw,
                    a_scale[m0 + fold_row[2], kt] * sw,
                    a_scale[m0 + fold_row[3], kt] * sw,
                )

                tCrC_raw.fill(0.0)
                for kb in cutlass.range_constexpr(FUSED_NUM_K_BLOCK):
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_cv[None, None, kb, pipe_read],
                        tCrA_cv[None, None, kb],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_cv[None, None, kb, pipe_read],
                        tCrB_cv[None, None, kb],
                    )
                    cute.gemm(
                        tiled_mma,
                        tCrC_raw,
                        tCrA[None, None, kb],
                        tCrB[None, None, kb],
                        tCrC_raw,
                    )

                for i in cutlass.range_constexpr(cute.size(tCrC)):
                    tCrC[i] = tCrC[i] + tCrC_raw[i] * rs[FOLD_CLASS_IDX[i]]

                cute.arch.sync_threads()
                pipe_read = (kt + 1) % GEMM_STAGES

            # ---- Fused epilogue: v = SiLU(gate)*up, requant, store --------
            # The pipeline smem is free now; alias it as the v tile plus the
            # reduction scratch (single SmemAllocator, overlay reinterpretation,
            # bitwise-verified technique).
            @cute.struct
            class EpiStorage:
                v: cute.struct.Align[
                    cute.struct.MemRange[cutlass.BFloat16, BM * BN], 16
                ]
                part: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, FUSED_UNITS], 16
                ]
                scl: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, BM], 16
                ]

            epi = EpiStorage(storage)
            v_s = epi.v.get_tensor(cute.make_layout((BM, BN), stride=(BN, 1)))
            s_part = epi.part.get_tensor(cute.make_layout(FUSED_UNITS))
            s_scl = epi.scl.get_tensor(cute.make_layout(BM))
            lim = tile_limit[mb]

            # SiLU(gate) * up in the reference's rounding order (bfloat16
            # after the SiLU and after the product), staging v in smem for
            # the exact per-row-group amax reduction silu_quant performed.
            for i in cutlass.range_constexpr(FUSED_ACC):
                row = tCcI[i][0]
                col = tCcI[i][1]
                gf = tCrC_gate_bf[i].to(cutlass.Float32)
                uf = tCrC[i].to(cutlass.Float32)
                sig = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(cutlass.Float32(0.0) - gf)
                )
                sf = (gf * sig).to(cutlass.BFloat16).to(cutlass.Float32)
                v_s[row, col] = (sf * uf).to(cutlass.BFloat16)
            cute.arch.sync_threads()

            # Per-128-group amax: one unit == (row, 16-element sub-block);
            # each row's 128 elements == 8 units, reduced by the unit-leader
            # exactly as silu_quant reduced its 8 sub-thread partials.
            for j in cutlass.range_constexpr(FUSED_UNITS // GEMM_THREADS):
                unit = tidx + j * GEMM_THREADS
                row = unit // SUBS_PER_GROUP
                sub = unit % SUBS_PER_GROUP
                amax = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(SUB_ELEMS):
                    vv = v_s[row, sub * SUB_ELEMS + i].to(cutlass.Float32)
                    amax = cute.max(amax, cute.abs(vv))
                s_part[unit] = amax
            cute.arch.sync_threads()

            if tidx % SUBS_PER_GROUP == 0:
                for j in cutlass.range_constexpr(FUSED_UNITS // GEMM_THREADS):
                    unit = tidx + j * GEMM_THREADS
                    row = unit // SUBS_PER_GROUP
                    m = s_part[unit]
                    for k in cutlass.range_constexpr(SUBS_PER_GROUP - 1):
                        m = cute.max(m, s_part[unit + 1 + k])
                    scale = cute.max(m, cutlass.Float32(1e-12)) / cutlass.Float32(FP8_MAX)
                    s_scl[row] = scale
                    if row < lim:
                        inter_s[(m0 + row) * (INTER // GROUP) + nb] = scale
            cute.arch.sync_threads()

            for j in cutlass.range_constexpr(FUSED_UNITS // GEMM_THREADS):
                unit = tidx + j * GEMM_THREADS
                row = unit // SUBS_PER_GROUP
                sub = unit % SUBS_PER_GROUP
                scale = s_scl[row]
                fq = cute.make_fragment_like(cute.make_layout(SUB_ELEMS), F8)
                for i in cutlass.range_constexpr(SUB_ELEMS):
                    vv = v_s[row, sub * SUB_ELEMS + i].to(cutlass.Float32)
                    qv = vv / scale
                    qv = cute.clamp(qv, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX))
                    fq[i] = qv.to(F8)
                if row < lim:
                    o16 = cute.make_tensor(
                        inter_q + (m0 + row) * INTER + nb * GROUP + sub * SUB_ELEMS,
                        cute.make_layout(SUB_ELEMS),
                    )
                    cute.autovec_copy(fq, o16)

else:

    @cute.kernel
    def gemm1_fused_kernel(  # noqa: F811
        mA: cute.Tensor,            # float8_e4m3fn (pad_cap, K1), position order
        a_scale: cute.Tensor,       # float32 (pad_cap, K1 / GROUP), position order
        block_expert: cute.Tensor,  # int32 [grid_m_cap]
        tile_limit: cute.Tensor,    # int32 [grid_m_cap]: valid rows per M tile
        mW: cute.Tensor,            # float8_e4m3fn (E, N1, K1)
        w_scale: cute.Pointer,      # float32 [E, N1/GROUP, K1/GROUP]
        inter_q: cute.Pointer,      # float8_e4m3fn [pad_cap, INTER], output,
                                    # position order
        inter_s: cute.Pointer,      # float32 [pad_cap, INTER/GROUP], output,
                                    # position order
        K: cutlass.Int32,           # K1
        k_groups: cutlass.Int32,    # K1 / GROUP
        n_groups: cutlass.Int32,    # N1 / GROUP (w_scale stride; 2x pair count)
        num_m: cutlass.Int32,       # grid_m_pad (number of M tiles)
        sA_layout: cute.Layout,
        sA_swz: cute.Swizzle,
        sB_layout: cute.Layout,
        sB_swz: cute.Swizzle,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        # Smem-staged fallback variant (FUSED_BK == GROUP/2): pipeline smem
        # shrinks to 24KB, so a dedicated 16KB bf16 gate-stage region
        # coexists with it (40KB total keeps 2 CTAs/SM on the 100KB/SM
        # device). Two k tiles == one scale group: both half-tiles accumulate
        # into tCrC_raw and the fold runs once per group (bitwise the same
        # products in the same K order).
        tidx, _, _ = cute.arch.thread_idx()
        L, _, _ = cute.arch.block_idx()

        num_n = n_groups // 2
        group_tiles = RASTER_GROUP_M * num_n
        group = L // group_tiles
        r = L % group_tiles
        mb = group * RASTER_GROUP_M + r % RASTER_GROUP_M
        nb = r // RASTER_GROUP_M

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[F8, cute.cosize(sA_layout)], 16
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[F8, cute.cosize(sB_layout)], 16
            ]
            g: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, BM * BN], 16
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
        sA = SharedStorage(storage).a.get_tensor(sA_layout, swizzle=sA_swz)
        sB = SharedStorage(storage).b.get_tensor(sB_layout, swizzle=sB_swz)
        sGate = SharedStorage(storage).g.get_tensor(
            cute.make_layout((BM, BN), stride=(BN, 1))
        )

        e = block_expert[mb]
        if e >= 0:
            m0 = mb * BM
            k_tiles = K // FUSED_BK
            n_blk_g = nb                        # weight block of the gate half
            n_blk_u = nb + INTER // GROUP       # weight block of the up half

            gW = mW[e, None, None]
            gA_mk = cute.local_tile(mA, (BM, FUSED_BK), (mb, None))
            gBg_nk = cute.local_tile(gW, (BN, FUSED_BK), (nb, None))
            gBu_nk = cute.local_tile(gW, (BN, FUSED_BK), (n_blk_u, None))

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA_mk)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gBg_nk)
            tBuB = thr_copy_B.partition_S(gBu_nk)
            tBsB = thr_copy_B.partition_D(sB)

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            cI = cute.make_identity_tensor((BM, BN))
            tCcI = thr_mma.partition_C(cI)
            tCrC = tiled_mma.make_fragment_C(tCcI.layout)
            tCrC_raw = cute.make_fragment_like(tCrC)
            tCrC.fill(0.0)

            atom_s2r = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x16x8bOp(False, 4), F8
            )
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r, tiled_mma)
            thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_cv = thr_s2r_A.partition_S(sA)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrA_cv = thr_s2r_A.retile(tCrA)
            tCsB_cv = thr_s2r_B.partition_S(sB)
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrB_cv = thr_s2r_B.retile(tCrB)

            fold_row = (tCcI[0][0], tCcI[2][0], tCcI[4][0], tCcI[6][0])

            # ---- gate K sweep (fold once per scale group = 2 half-tiles) ---
            cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
            cute.copy(tiled_copy_B, tBgB[None, None, None, 0], tBsB[None, None, None, 0])
            cute.arch.cp_async_commit_group()

            pipe_read = cutlass.Int32(0)
            for kg in cutlass.range(k_groups):
                # First half-tile (2*kg) is already in flight; issue the
                # second half-tile (2*kg + 1), then compute the first.
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, 2 * kg + 1],
                    tAsA[None, None, None, (2 * kg + 1) % GEMM_STAGES],
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, 2 * kg + 1],
                    tBsB[None, None, None, (2 * kg + 1) % GEMM_STAGES],
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()

                sw = w_scale[e * n_groups * k_groups + n_blk_g * k_groups + kg]
                rs = (
                    a_scale[m0 + fold_row[0], kg] * sw,
                    a_scale[m0 + fold_row[1], kg] * sw,
                    a_scale[m0 + fold_row[2], kg] * sw,
                    a_scale[m0 + fold_row[3], kg] * sw,
                )

                tCrC_raw.fill(0.0)
                for kb in cutlass.range_constexpr(FUSED_NUM_K_BLOCK):
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_cv[None, None, kb, pipe_read],
                        tCrA_cv[None, None, kb],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_cv[None, None, kb, pipe_read],
                        tCrB_cv[None, None, kb],
                    )
                    cute.gemm(
                        tiled_mma,
                        tCrC_raw,
                        tCrA[None, None, kb],
                        tCrB[None, None, kb],
                        tCrC_raw,
                    )

                # Pipeline hazard barrier: all threads must have finished
                # reading stage (2*kg)%2 (first half-tile) before the issue
                # below overwrites it with tile 2*kg+2.
                cute.arch.sync_threads()

                # Issue the first half-tile of the next group (if any), then
                # compute the second half-tile of this group.
                if 2 * kg + 2 < k_tiles:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, 2 * kg + 2],
                        tAsA[None, None, None, (2 * kg + 2) % GEMM_STAGES],
                    )
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, 2 * kg + 2],
                        tBsB[None, None, None, (2 * kg + 2) % GEMM_STAGES],
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                for kb in cutlass.range_constexpr(FUSED_NUM_K_BLOCK):
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_cv[None, None, kb, (2 * kg + 1) % GEMM_STAGES],
                        tCrA_cv[None, None, kb],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_cv[None, None, kb, (2 * kg + 1) % GEMM_STAGES],
                        tCrB_cv[None, None, kb],
                    )
                    cute.gemm(
                        tiled_mma,
                        tCrC_raw,
                        tCrA[None, None, kb],
                        tCrB[None, None, kb],
                        tCrC_raw,
                    )

                for i in cutlass.range_constexpr(cute.size(tCrC)):
                    tCrC[i] = tCrC[i] + tCrC_raw[i] * rs[FOLD_CLASS_IDX[i]]

                cute.arch.sync_threads()
                pipe_read = (2 * kg + 2) % GEMM_STAGES

            # Stage the gate accumulator's bf16-rounded values (the exact
            # bytes the old c1 store materialized) into the dedicated smem
            # region; the up sweep's first barrier makes them visible.
            for i in cutlass.range_constexpr(FUSED_ACC):
                sGate[tCcI[i][0], tCcI[i][1]] = tCrC[i].to(cutlass.BFloat16)

            # ---- up K sweep ------------------------------------------------
            tCrC.fill(0.0)
            cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
            cute.copy(tiled_copy_B, tBuB[None, None, None, 0], tBsB[None, None, None, 0])
            cute.arch.cp_async_commit_group()

            pipe_read = cutlass.Int32(0)
            for kg in cutlass.range(k_groups):
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, 2 * kg + 1],
                    tAsA[None, None, None, (2 * kg + 1) % GEMM_STAGES],
                )
                cute.copy(
                    tiled_copy_B,
                    tBuB[None, None, None, 2 * kg + 1],
                    tBsB[None, None, None, (2 * kg + 1) % GEMM_STAGES],
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()

                sw = w_scale[e * n_groups * k_groups + n_blk_u * k_groups + kg]
                rs = (
                    a_scale[m0 + fold_row[0], kg] * sw,
                    a_scale[m0 + fold_row[1], kg] * sw,
                    a_scale[m0 + fold_row[2], kg] * sw,
                    a_scale[m0 + fold_row[3], kg] * sw,
                )

                tCrC_raw.fill(0.0)
                for kb in cutlass.range_constexpr(FUSED_NUM_K_BLOCK):
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_cv[None, None, kb, pipe_read],
                        tCrA_cv[None, None, kb],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_cv[None, None, kb, pipe_read],
                        tCrB_cv[None, None, kb],
                    )
                    cute.gemm(
                        tiled_mma,
                        tCrC_raw,
                        tCrA[None, None, kb],
                        tCrB[None, None, kb],
                        tCrC_raw,
                    )

                cute.arch.sync_threads()

                if 2 * kg + 2 < k_tiles:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, 2 * kg + 2],
                        tAsA[None, None, None, (2 * kg + 2) % GEMM_STAGES],
                    )
                    cute.copy(
                        tiled_copy_B,
                        tBuB[None, None, None, 2 * kg + 2],
                        tBsB[None, None, None, (2 * kg + 2) % GEMM_STAGES],
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                for kb in cutlass.range_constexpr(FUSED_NUM_K_BLOCK):
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_cv[None, None, kb, (2 * kg + 1) % GEMM_STAGES],
                        tCrA_cv[None, None, kb],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_cv[None, None, kb, (2 * kg + 1) % GEMM_STAGES],
                        tCrB_cv[None, None, kb],
                    )
                    cute.gemm(
                        tiled_mma,
                        tCrC_raw,
                        tCrA[None, None, kb],
                        tCrB[None, None, kb],
                        tCrC_raw,
                    )

                for i in cutlass.range_constexpr(cute.size(tCrC)):
                    tCrC[i] = tCrC[i] + tCrC_raw[i] * rs[FOLD_CLASS_IDX[i]]

                cute.arch.sync_threads()
                pipe_read = (2 * kg + 2) % GEMM_STAGES

            # ---- Fused epilogue: v = SiLU(gate)*up, requant, store --------
            # The pipeline smem is free now; alias it as the v tile plus the
            # reduction scratch (the sGate region sits above the alias window
            # and is read for gf in the first loop below).
            @cute.struct
            class EpiStorage:
                v: cute.struct.Align[
                    cute.struct.MemRange[cutlass.BFloat16, BM * BN], 16
                ]
                part: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, FUSED_UNITS], 16
                ]
                scl: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, BM], 16
                ]

            epi = EpiStorage(storage)
            v_s = epi.v.get_tensor(cute.make_layout((BM, BN), stride=(BN, 1)))
            s_part = epi.part.get_tensor(cute.make_layout(FUSED_UNITS))
            s_scl = epi.scl.get_tensor(cute.make_layout(BM))
            lim = tile_limit[mb]

            for i in cutlass.range_constexpr(FUSED_ACC):
                row = tCcI[i][0]
                col = tCcI[i][1]
                gf = sGate[row, col].to(cutlass.Float32)
                uf = tCrC[i].to(cutlass.Float32)
                sig = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(cutlass.Float32(0.0) - gf)
                )
                sf = (gf * sig).to(cutlass.BFloat16).to(cutlass.Float32)
                v_s[row, col] = (sf * uf).to(cutlass.BFloat16)
            cute.arch.sync_threads()

            for j in cutlass.range_constexpr(FUSED_UNITS // GEMM_THREADS):
                unit = tidx + j * GEMM_THREADS
                row = unit // SUBS_PER_GROUP
                sub = unit % SUBS_PER_GROUP
                amax = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(SUB_ELEMS):
                    vv = v_s[row, sub * SUB_ELEMS + i].to(cutlass.Float32)
                    amax = cute.max(amax, cute.abs(vv))
                s_part[unit] = amax
            cute.arch.sync_threads()

            if tidx % SUBS_PER_GROUP == 0:
                for j in cutlass.range_constexpr(FUSED_UNITS // GEMM_THREADS):
                    unit = tidx + j * GEMM_THREADS
                    row = unit // SUBS_PER_GROUP
                    m = s_part[unit]
                    for k in cutlass.range_constexpr(SUBS_PER_GROUP - 1):
                        m = cute.max(m, s_part[unit + 1 + k])
                    scale = cute.max(m, cutlass.Float32(1e-12)) / cutlass.Float32(FP8_MAX)
                    s_scl[row] = scale
                    if row < lim:
                        inter_s[(m0 + row) * (INTER // GROUP) + nb] = scale
            cute.arch.sync_threads()

            for j in cutlass.range_constexpr(FUSED_UNITS // GEMM_THREADS):
                unit = tidx + j * GEMM_THREADS
                row = unit // SUBS_PER_GROUP
                sub = unit % SUBS_PER_GROUP
                scale = s_scl[row]
                fq = cute.make_fragment_like(cute.make_layout(SUB_ELEMS), F8)
                for i in cutlass.range_constexpr(SUB_ELEMS):
                    vv = v_s[row, sub * SUB_ELEMS + i].to(cutlass.Float32)
                    qv = vv / scale
                    qv = cute.clamp(qv, cutlass.Float32(-FP8_MAX), cutlass.Float32(FP8_MAX))
                    fq[i] = qv.to(F8)
                if row < lim:
                    o16 = cute.make_tensor(
                        inter_q + (m0 + row) * INTER + nb * GROUP + sub * SUB_ELEMS,
                        cute.make_layout(SUB_ELEMS),
                    )
                    cute.autovec_copy(fq, o16)


@cute.kernel
def gemm_tc_kernel(
    mA: cute.Tensor,            # float8_e4m3fn (pad_cap, K), position order
    a_scale: cute.Tensor,       # float32 (pad_cap, K / GROUP), position order
    block_expert: cute.Tensor,  # int32 [grid_m_cap]
    tile_limit: cute.Tensor,    # int32 [grid_m_cap]: valid rows per M tile
    mW: cute.Tensor,            # float8_e4m3fn (E, N, K)
    w_scale: cute.Pointer,      # float32 [E, N/GROUP, K/GROUP]
    mC: cute.Tensor,            # bfloat16 (pad_cap, N), output, position order
    K: cutlass.Int32,
    k_groups: cutlass.Int32,    # K / GROUP
    n_groups: cutlass.Int32,    # N / GROUP
    num_m: cutlass.Int32,       # grid_m_cap (number of M tiles)
    sA_layout: cute.Layout,
    sA_swz: cute.Swizzle,
    sB_layout: cute.Layout,
    sB_swz: cute.Swizzle,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    L, _, _ = cute.arch.block_idx()

    # L2-grouped rasterization (branch-free). CTAs are launched in a flat 1-D
    # linear order and remapped bijectively onto (mb, nb) so that a group of
    # RASTER_GROUP_M consecutive M tiles sweeps ALL N tiles before advancing.
    # The shared A row block of a group then stays L2-resident across the whole
    # N-tile sweep, instead of being re-streamed from DRAM once per N-tile
    # column as under mb-fastest order. The host pads num_m up to a multiple of
    # RASTER_GROUP_M (extra block_expert entries are sentinel -1 and early-exit),
    # so there is no partial group and this is pure arithmetic with no dynamic
    # branch. Per-tile arithmetic is unchanged (each output tile is still
    # computed exactly once, in the same K order), so results are bit-identical.
    # num_n == n_groups because BN == GROUP == 128.
    num_n = n_groups
    group_tiles = RASTER_GROUP_M * num_n
    group = L // group_tiles
    r = L % group_tiles
    mb = group * RASTER_GROUP_M + r % RASTER_GROUP_M
    nb = r // RASTER_GROUP_M

    @cute.struct
    class SharedStorage:
        a: cute.struct.Align[
            cute.struct.MemRange[F8, cute.cosize(sA_layout)], 16
        ]
        b: cute.struct.Align[
            cute.struct.MemRange[F8, cute.cosize(sB_layout)], 16
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
    sA = SharedStorage(storage).a.get_tensor(sA_layout, swizzle=sA_swz)
    sB = SharedStorage(storage).b.get_tensor(sB_layout, swizzle=sB_swz)

    e = block_expert[mb]
    if e >= 0:
        m0 = mb * BM
        n0 = nb * BN
        k_tiles = K // BK
        n_blk = n0 // GROUP

        # gmem tiles: A rows are dense sorted positions; B is the expert's
        # weight matrix slice.
        gW = mW[e, None, None]
        gA_mk = cute.local_tile(mA, (BM, BK), (mb, None))
        gB_nk = cute.local_tile(gW, (BN, BK), (nb, None))
        gC_mn = cute.local_tile(mC, (BM, BN), (mb, nb))

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA_mk)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB_nk)
        tBsB = thr_copy_B.partition_D(sB)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC_mn)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC_raw = cute.make_fragment_like(tCrC)
        tCrC.fill(0.0)

        atom_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x16x8bOp(False, 4), F8
        )
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r, tiled_mma)
        thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
        thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsA_cv = thr_s2r_A.partition_S(sA)
        tCrA_cv = thr_s2r_A.retile(tCrA)
        tCsB_cv = thr_s2r_B.partition_S(sB)
        tCrB_cv = thr_s2r_B.retile(tCrB)

        # Identity tensor gives each accumulator element its (m, n) coordinate
        # inside the CTA tile for the scale-fold row lookup.
        cI = cute.make_identity_tensor((BM, BN))
        tCcI = thr_mma.partition_C(cI)

        # Register-fold: the 4 row classes of the C fragment (FOLD_CLASS) are
        # represented by elements 0, 2, 4, 6; capture their runtime M rows once.
        fold_row = (tCcI[0][0], tCcI[2][0], tCcI[4][0], tCcI[6][0])

        # Prologue: issue the first k tile.
        cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
        cute.copy(tiled_copy_B, tBgB[None, None, None, 0], tBsB[None, None, None, 0])
        cute.arch.cp_async_commit_group()

        pipe_read = cutlass.Int32(0)
        for kt in cutlass.range(k_tiles):
            # Issue tile kt+1 into the other stage, then wait for tile kt.
            if kt + 1 < k_tiles:
                wr = (kt + 1) % GEMM_STAGES
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, kt + 1],
                    tAsA[None, None, None, wr],
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, kt + 1],
                    tBsB[None, None, None, wr],
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # Register-fold scales, issued right after the barrier so the L1
            # loads overlap the ldmatrix+mma phase instead of stalling at the
            # fold. Each thread's 64 accumulator elements touch only the 4 M
            # rows in fold_row (static class map FOLD_CLASS, probe-verified
            # for all 128 threads); the arithmetic is bitwise-identical to the
            # incumbent's smem-staged fold (same single a_scale*sw rounding,
            # same raw*rs FMA per element, same element order).
            sw = w_scale[e * n_groups * k_groups + n_blk * k_groups + kt]
            rs = (
                a_scale[m0 + fold_row[0], kt] * sw,
                a_scale[m0 + fold_row[1], kt] * sw,
                a_scale[m0 + fold_row[2], kt] * sw,
                a_scale[m0 + fold_row[3], kt] * sw,
            )

            # This k tile is exactly one 128-K scale group: accumulate the raw
            # FP8 products of its 4 mma-k-blocks into tCrC_raw.
            tCrC_raw.fill(0.0)
            for kb in cutlass.range_constexpr(NUM_K_BLOCK):
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_cv[None, None, kb, pipe_read],
                    tCrA_cv[None, None, kb],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_cv[None, None, kb, pipe_read],
                    tCrB_cv[None, None, kb],
                )
                cute.gemm(
                    tiled_mma,
                    tCrC_raw,
                    tCrA[None, None, kb],
                    tCrB[None, None, kb],
                    tCrC_raw,
                )

            # Fold: main += raw * rs[class(i)], with rs issued above.
            for i in cutlass.range_constexpr(cute.size(tCrC)):
                tCrC[i] = tCrC[i] + tCrC_raw[i] * rs[FOLD_CLASS_IDX[i]]

            # Pipeline hazard barrier: all threads must have finished the
            # ldmatrix reads of this iteration's smem stage before any thread
            # wraps around and issues the cp.async that overwrites it two
            # iterations later (stage (kt+1)%2 was last read at iteration
            # kt-1). The incumbent's fold barrier served this role too.
            cute.arch.sync_threads()

            pipe_read = (kt + 1) % GEMM_STAGES

        # Epilogue: bf16 store of the tile's VALID rows only. Rows past
        # tile_limit[mb] are this expert's BM-rounding padding: computed by
        # the tile-granular MMA but never read by any downstream stage (silu
        # and combine index real positions through pos_of_pair), so skipping
        # their stores removes never-read DRAM write traffic while leaving
        # every stored value unchanged. Sentinel tiles early-exited above.
        lim = tile_limit[mb]
        for i in cutlass.range_constexpr(cute.size(tCrC)):
            if tCcI[i][0] < lim:
                mC[m0 + tCcI[i][0], n0 + tCcI[i][1]] = tCrC[i].to(
                    cutlass.BFloat16
                )


@cute.kernel
def combine_kernel(
    o2: cute.Pointer,           # bfloat16 [pad_cap, HIDDEN], position order
    pos_of_pair: cute.Tensor,   # int32 [P]
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
            acc = acc + topk_w[pair] * o2[pos_of_pair[pair] * HIDDEN + h].to(
                cutlass.Float32
            )
        out[tid] = acc.to(cutlass.BFloat16)


# ---------------------------------------------------------------------------
# Host-side pipeline (one @cute.jit launcher; compiled once per Model)
# ---------------------------------------------------------------------------
@cute.jit
def moe_pipeline(
    hidden: cute.Pointer,       # bfloat16 [T * HIDDEN]
    topk_w: cute.Tensor,        # float32 [T * TOPK]
    topk_ids: cute.Tensor,      # int32 [T * TOPK]
    w1: cute.Pointer,           # float8_e4m3fn [E, N1, K1]
    w1_scale: cute.Pointer,     # float32 [E, N1/GROUP, K1/GROUP]
    w2: cute.Pointer,           # float8_e4m3fn [E, N2, K2]
    w2_scale: cute.Pointer,     # float32 [E, N2/GROUP, K2/GROUP]
    hidden_q_perm: cute.Pointer,  # float8_e4m3fn [pad_cap, HIDDEN]
    hidden_s_perm: cute.Pointer,  # float32 [pad_cap, HIDDEN/GROUP]
    sorted_ids: cute.Tensor,    # int32 [pad_cap]
    block_expert: cute.Tensor,  # int32 [grid_m_cap]
    tile_limit: cute.Tensor,    # int32 [grid_m_cap]: valid rows per M tile
    pos_of_pair: cute.Tensor,   # int32 [P]
    rbuf: cute.Pointer,         # int32 [2 * NUM_EXPERTS] routing bookkeeping
    inter_q: cute.Pointer,      # float8_e4m3fn [pad_cap, INTER]
    inter_s: cute.Pointer,      # float32 [pad_cap, INTER/GROUP]
    o2: cute.Pointer,           # bfloat16 [pad_cap * N2]
    out: cute.Tensor,           # bfloat16 [T, HIDDEN]
    T: cutlass.Int32,
):
    P = T * TOPK
    pad_cap = P + NUM_EXPERTS * BM + BM
    grid_m_cap = pad_cap // BM
    # Pad the M-tile count up to a multiple of RASTER_GROUP_M so the grouped
    # rasterization in the GEMM kernels is branch-free (no partial last
    # group). Extra M tiles carry block_expert == -1 (sentinel) and early-exit.
    grid_m_pad = cute.ceil_div(grid_m_cap, RASTER_GROUP_M) * RASTER_GROUP_M

    # Routing: init/fill -> atomic histogram -> 256-entry scan -> atomic
    # scatter. Sequential launches on one stream order the phases.
    n_init = pad_cap + grid_m_pad + 2 * NUM_EXPERTS
    routing_init_kernel(sorted_ids, block_expert, rbuf, pad_cap, grid_m_pad).launch(
        grid=[cute.ceil_div(n_init, ELEM_THREADS), 1, 1],
        block=[ELEM_THREADS, 1, 1],
    )
    routing_hist_kernel(topk_ids, rbuf, P).launch(
        grid=[cute.ceil_div(P, ELEM_THREADS), 1, 1],
        block=[ELEM_THREADS, 1, 1],
    )
    routing_scan_kernel(rbuf, block_expert, tile_limit).launch(
        grid=[1, 1, 1], block=[NUM_EXPERTS, 1, 1]
    )
    routing_place_kernel(topk_ids, sorted_ids, pos_of_pair, rbuf, P).launch(
        grid=[cute.ceil_div(P, ELEM_THREADS), 1, 1],
        block=[ELEM_THREADS, 1, 1],
    )

    # Fused quantization + gather: two blocks per token (exact grid), each
    # writing half of the token's pair positions; bytes land directly in
    # sorted position order.
    quant_gather_kernel(hidden, pos_of_pair, hidden_q_perm, hidden_s_perm).launch(
        grid=[2 * T, 1, 1],
        block=[QUANT_THREADS, 1, 1],
    )

    # ---- GEMM 1 (gate_up) fused with SiLU + requant -------------------------
    # Pair-CTAs over (positions, gate/up 128-col pairs); stores inter_q /
    # inter_s directly, deleting the c1 round-trip and the silu_quant kernel.
    mA1 = cute.make_tensor(
        hidden_q_perm, cute.make_layout((pad_cap, K1), stride=(K1, 1))
    )
    a_scale1 = cute.make_tensor(
        hidden_s_perm,
        cute.make_layout((pad_cap, K1 // GROUP), stride=(K1 // GROUP, 1)),
    )
    mW1 = cute.make_tensor(
        w1,
        cute.make_layout(
            (NUM_EXPERTS, N1, K1), stride=(N1 * K1, K1, 1)
        ),
    )
    sA1_layout, sA1_swz = _make_smem_layout(BM, FUSED_BK, GEMM_STAGES)
    sB1_layout, sB1_swz = _make_smem_layout(BN, FUSED_BK, GEMM_STAGES)
    tiled_copy_A1 = _make_gmem_tiled_copy(GEMM_THREADS, FUSED_BK)
    tiled_copy_B1 = _make_gmem_tiled_copy(GEMM_THREADS, FUSED_BK)
    tiled_mma1 = _make_tiled_mma_fp8()
    gemm1_fused_kernel(
        mA1,
        a_scale1,
        block_expert,
        tile_limit,
        mW1,
        w1_scale,
        inter_q,
        inter_s,
        K1,
        K1 // GROUP,
        N1 // GROUP,
        grid_m_pad,
        sA1_layout,
        sA1_swz,
        sB1_layout,
        sB1_swz,
        tiled_copy_A1,
        tiled_copy_B1,
        tiled_mma1,
    ).launch(
        grid=[grid_m_pad * (INTER // BN), 1, 1], block=[GEMM_THREADS, 1, 1]
    )

    # ---- GEMM 2 (down): C1[BM x BN] tiles over (positions, N2) -------------
    mA2 = cute.make_tensor(
        inter_q, cute.make_layout((pad_cap, K2), stride=(K2, 1))
    )
    a_scale2 = cute.make_tensor(
        inter_s,
        cute.make_layout((pad_cap, K2 // GROUP), stride=(K2 // GROUP, 1)),
    )
    mW2 = cute.make_tensor(
        w2,
        cute.make_layout(
            (NUM_EXPERTS, N2, K2), stride=(N2 * K2, K2, 1)
        ),
    )
    mC2 = cute.make_tensor(
        o2, cute.make_layout((pad_cap, N2), stride=(N2, 1))
    )
    sA2_layout, sA2_swz = _make_smem_layout(BM, BK, GEMM_STAGES)
    sB2_layout, sB2_swz = _make_smem_layout(BN, BK, GEMM_STAGES)
    tiled_copy_A2 = _make_gmem_tiled_copy(GEMM_THREADS, BK)
    tiled_copy_B2 = _make_gmem_tiled_copy(GEMM_THREADS, BK)
    tiled_mma2 = _make_tiled_mma_fp8()
    gemm_tc_kernel(
        mA2,
        a_scale2,
        block_expert,
        tile_limit,
        mW2,
        w2_scale,
        mC2,
        K2,
        K2 // GROUP,
        N2 // GROUP,
        grid_m_pad,
        sA2_layout,
        sA2_swz,
        sB2_layout,
        sB2_swz,
        tiled_copy_A2,
        tiled_copy_B2,
        tiled_mma2,
    ).launch(
        grid=[grid_m_pad * (N2 // BN), 1, 1], block=[GEMM_THREADS, 1, 1]
    )

    combine_kernel(o2, pos_of_pair, topk_w, out, T * HIDDEN).launch(
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
        grid_m_pad0 = -(-grid_m_cap0 // RASTER_GROUP_M) * RASTER_GROUP_M

        def t16(n: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(n, device=device, dtype=dtype)

        f = lambda tensor: from_dlpack(tensor, assumed_align=16)  # noqa: E731

        def fptr(tensor, dtype) -> cutlass.Pointer:
            return make_ptr(dtype, tensor.data_ptr(), _gmem, assumed_align=16)

        hidden = fptr(t16(t0 * HIDDEN, torch.bfloat16), cutlass.BFloat16)
        topk_w = f(t16(t0 * TOPK, torch.float32))
        topk_ids = f(t16(t0 * TOPK, torch.int32))
        w1 = fptr(t16(1, torch.float8_e4m3fn), cutlass.Float8E4M3FN)
        w1_scale = fptr(t16(1, torch.float32), cutlass.Float32)
        w2 = fptr(t16(1, torch.float8_e4m3fn), cutlass.Float8E4M3FN)
        w2_scale = fptr(t16(1, torch.float32), cutlass.Float32)
        hidden_q_perm = fptr(t16(1, torch.float8_e4m3fn), cutlass.Float8E4M3FN)
        hidden_s_perm = fptr(t16(1, torch.float32), cutlass.Float32)
        sorted_ids = f(t16(pad_cap0, torch.int32))
        block_expert = f(t16(grid_m_pad0, torch.int32))
        tile_limit = f(t16(grid_m_pad0, torch.int32))
        pos_of_pair = f(t16(p0, torch.int32))
        rbuf = fptr(t16(2 * NUM_EXPERTS, torch.int32), cutlass.Int32)
        inter_q = fptr(t16(1, torch.float8_e4m3fn), cutlass.Float8E4M3FN)
        inter_s = fptr(t16(1, torch.float32), cutlass.Float32)
        o2 = fptr(t16(1, torch.bfloat16), cutlass.BFloat16)
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
            hidden_q_perm,
            hidden_s_perm,
            sorted_ids,
            block_expert,
            tile_limit,
            pos_of_pair,
            rbuf,
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
        grid_m_pad = -(-grid_m_cap // RASTER_GROUP_M) * RASTER_GROUP_M

        hidden_states = hidden_states.contiguous()
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        w1_scale = w1_scale.contiguous()
        w2_scale = w2_scale.contiguous()

        hidden_q_perm = torch.empty(pad_cap, HIDDEN, device=device, dtype=torch.float8_e4m3fn)
        hidden_s_perm = torch.empty(pad_cap, HIDDEN // GROUP, device=device, dtype=torch.float32)
        sorted_ids = torch.empty(pad_cap, device=device, dtype=torch.int32)
        block_expert = torch.empty(grid_m_pad, device=device, dtype=torch.int32)
        tile_limit = torch.empty(grid_m_pad, device=device, dtype=torch.int32)
        pos_of_pair = torch.empty(pairs, device=device, dtype=torch.int32)
        rbuf = torch.empty(2 * NUM_EXPERTS, device=device, dtype=torch.int32)
        inter_q = torch.empty(pad_cap, INTER, device=device, dtype=torch.float8_e4m3fn)
        inter_s = torch.empty(pad_cap, INTER // GROUP, device=device, dtype=torch.float32)
        o2 = torch.empty(pad_cap, N2, device=device, dtype=torch.bfloat16)
        out = torch.empty(token_count, HIDDEN, device=device, dtype=torch.bfloat16)

        def f(tensor: torch.Tensor):
            return from_dlpack(tensor.view(-1), assumed_align=16)

        def fptr(tensor: torch.Tensor, dtype):
            return make_ptr(dtype, tensor.data_ptr(), _gmem, assumed_align=16)

        self._executor(
            fptr(hidden_states, cutlass.BFloat16),
            f(topk_weights),
            f(topk_ids),
            fptr(w1, cutlass.Float8E4M3FN),
            fptr(w1_scale, cutlass.Float32),
            fptr(w2, cutlass.Float8E4M3FN),
            fptr(w2_scale, cutlass.Float32),
            fptr(hidden_q_perm, cutlass.Float8E4M3FN),
            fptr(hidden_s_perm, cutlass.Float32),
            f(sorted_ids),
            f(block_expert),
            f(tile_limit),
            f(pos_of_pair),
            fptr(rbuf, cutlass.Int32),
            fptr(inter_q, cutlass.Float8E4M3FN),
            fptr(inter_s, cutlass.Float32),
            fptr(o2, cutlass.BFloat16),
            f(out),
            token_count,
        )
        return out
