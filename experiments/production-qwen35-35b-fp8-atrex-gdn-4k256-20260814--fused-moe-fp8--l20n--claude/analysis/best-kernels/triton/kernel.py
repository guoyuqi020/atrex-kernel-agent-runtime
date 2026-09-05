"""Block-scaled FP8 fused MoE implemented entirely in self-authored Triton kernels.

Every compute step of the reference operator is expressed as a Triton kernel:

1. ``_expert_count_kernel``  - histogram of flattened top-k expert ids.
2. ``_expert_scan_kernel``   - exclusive scan of per-expert token counts padded to
                               ``BLOCK_M`` so every GEMM tile belongs to one expert.
3. ``_token_scatter_kernel`` - gathers the (token, expert-slot) pairs into the
                               padded, expert-major ``sorted_token_ids`` layout.
3b. ``_activation_quant_kernel`` - quantizes every token row of ``hidden_states``
                               to FP8 per 128-element K group exactly like the
                               reference ``_quantize_activation``
                               (scale = max(amax, 1e-12)/448, RTNE cast), once
                               per token instead of once per routed slot.
4. ``_gemm_gate_kernel`` / ``_gemm_up_requant_kernel`` - the w1 grouped GEMM
                               split into two single-accumulator passes.  The
                               old fused dual-dot kept acc_g and acc_u live
                               simultaneously, pinning the kernel at 255
                               registers and one CTA per SM.  The gate pass
                               accumulates only acc_g and stores the bf16
                               silu(g) tile; the up pass accumulates only
                               acc_u, multiplies the stored silu(g) tile in,
                               and requantizes the intermediate exactly like
                               the former fused epilogue.  Both consume the
                               pre-quantized FP8 activations and per-group
                               scales (bit-identical to in-loop quantization)
                               and launch under a 128-register budget so two
                               CTAs reside per SM.
5. ``_gemm_down_kernel``     - grouped GEMM against ``w2`` with the stored
                               intermediate FP8 values/scales, FP32 accumulate.
                               The epilogue is fused with the top-k weighting:
                               each tile is rounded to bf16 exactly like the old
                               partial buffer, multiplied by its per-(token, slot)
                               top-k weight, and scattered with FP32 atomic adds
                               directly into the ``[token_count, hidden_size]``
                               output accumulator, removing the partial-buffer
                               round trip entirely.  The kernel always tiles at
                               BLOCK_M rows and launches under the same two-CTA
                               register budget as the gate/up passes; the
                               num_pairs sentinel check is retained as a
                               defensive skip of all-padding tiles.
6. ``_fp32_cast_kernel``     - casts the FP32 accumulated output to the
                               activation dtype.

The inter-pass buffers (``silu_g``, ``inter_q``, ``inter_scale``) are
indexed by sorted slot position -- the tile row index inside the padded
expert segments -- rather than by pair id.  Every GEMM tile's 64 rows are
therefore contiguous in those buffers, so the gate/up epilogue stores and
the down GEMM's activation operand loads are fully coalesced tile I/O
instead of 64-row scatters/gathers across pair-major memory.  The buffers
span the ``sorted_token_ids`` extent; positions past ``npost`` are never
touched (early-return beyond ``npost``, valid-masked padding rows).

``torch`` is used only for tensor allocation and shape bookkeeping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

# tl.constexpr globals so the values are visible inside @triton.jit kernels.
FP8_E4M3_MAX = tl.constexpr(448.0)  # == torch.finfo(torch.float8_e4m3fn).max
SCALE_EPS = tl.constexpr(1e-12)  # same amax floor as the reference _quantize_activation

BLOCK_M = 64  # rows per gate-up GEMM tile; also the down-GEMM tile width
# Every GEMM tiles at BLOCK_M rows: the 64-row tile keeps the fp32 accumulator
# in the 32-regs/thread class that fits the two-CTA register budget, while
# 128-row tiles spill under it.  Expert segments pad to BLOCK_M rows as well
# (see ``Model.forward``): that is the minimum padding compatible with
# single-expert 64-row tiles; coarser padding was pure waste once the down
# GEMM moved to 64-row tiles, and finer padding was falsified in Epoch 3.
BLOCK_N = 128  # matches block_shape[0]; keeps weight-scale blocks tile-aligned
BLOCK_K = 128  # matches block_shape[1]; activation quantization group width
DISPATCH_BLOCK = 1024  # lanes per routing histogram/scatter program
ACT_GROUPS = 8  # 128-element quantization groups per activation-prepass program


@triton.jit
def _expert_count_kernel(
    ids_ptr,
    counts_ptr,
    num_pairs,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_pairs
    ids = tl.load(ids_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + ids, 1, mask=mask)


@triton.jit
def _expert_scan_kernel(
    counts_ptr,
    start_ptr,
    npost_ptr,
    num_experts,
    BM: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < num_experts
    counts = tl.load(counts_ptr + offs_e, mask=mask_e, other=0)
    padded = ((counts + BM - 1) // BM) * BM
    inclusive = tl.cumsum(padded, axis=0)
    tl.store(start_ptr + offs_e, inclusive - padded, mask=mask_e)
    tl.store(npost_ptr, tl.sum(padded, axis=0))


@triton.jit
def _token_scatter_kernel(
    ids_ptr,
    start_ptr,
    cursor_ptr,
    sorted_ptr,
    num_pairs,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_pairs
    ids = tl.load(ids_ptr + offs, mask=mask, other=0)
    rank = tl.atomic_add(cursor_ptr + ids, 1, mask=mask)
    base = tl.load(start_ptr + ids, mask=mask, other=0)
    tl.store(sorted_ptr + base + rank, offs, mask=mask)


@triton.jit
def _activation_quant_kernel(
    hidden_ptr,
    q_ptr,
    scale_ptr,
    num_groups,
    GROUPS: tl.constexpr,
):
    # Quantize each 128-element K group of every token row exactly like the
    # reference _quantize_activation: scale = max(amax, 1e-12)/448, RTNE fp8.
    pid = tl.program_id(0)
    group = pid * GROUPS + tl.arange(0, GROUPS)
    mask_g = group < num_groups
    offs = group[:, None] * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(hidden_ptr + offs, mask=mask_g[:, None], other=0.0)
    xf = x.to(tl.float32)
    amax = tl.max(tl.abs(xf), axis=1)
    scale = tl.maximum(amax, SCALE_EPS) / FP8_E4M3_MAX
    q = xf / scale[:, None]
    q = tl.minimum(tl.maximum(q, -FP8_E4M3_MAX), FP8_E4M3_MAX)
    q = q.to(tl.float8e4nv)
    tl.store(q_ptr + offs, q, mask=mask_g[:, None])
    tl.store(scale_ptr + group, scale, mask=mask_g)


@triton.jit
def _gemm_gate_kernel(
    sorted_ptr,
    ids_ptr,
    act_q_ptr,
    act_scale_ptr,
    w1_ptr,
    w1_scale_ptr,
    silu_g_ptr,
    npost_ptr,
    num_pairs,
    hidden_size,
    inter_size,
    w1s_stride_e,
    w1s_stride_n,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BM
    if row_start >= tl.load(npost_ptr):
        return

    # Every tile inside the padded region starts on a real (token, slot) pair,
    # so the first entry identifies the tile's expert.  Segments pad to BM
    # rows, so every BM-tile holds at least one real pair; the sentinel check
    # is retained defensively: since each segment fills contiguously from its
    # start, a tile whose first entry is the num_pairs sentinel is all padding
    # -> skip it (no weight streaming).
    first_pair = tl.load(sorted_ptr + row_start)
    if first_pair >= num_pairs:
        return
    expert = tl.load(ids_ptr + first_pair)

    slots = row_start + tl.arange(0, BM)
    pair_ids = tl.load(sorted_ptr + slots)
    valid = pair_ids < num_pairs
    token = pair_ids // TOPK

    offs_n = pid_n * BN + tl.arange(0, BN)
    num_k_blocks = hidden_size // BK
    num_act_groups = hidden_size // GK

    # Single fp32 accumulator: the former fused dual-dot kept acc_g and acc_u
    # live across the whole K-loop, pinning the kernel at the 255-register
    # ceiling and one CTA per SM.
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    w1_base = w1_ptr + expert * (2 * inter_size) * hidden_size
    for kb in range(num_k_blocks):
        offs_k = kb * BK + tl.arange(0, BK)
        # Pre-quantized FP8 activations (once per token, bit-identical to the
        # old in-loop quantization) plus their per-128-group fp32 scales.
        aq = tl.load(
            act_q_ptr + token[:, None] * hidden_size + offs_k[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        a_scale = tl.load(
            act_scale_ptr + token * num_act_groups + kb,
            mask=valid,
            other=0.0,
        )
        w_gate = tl.load(
            w1_base + offs_n[None, :] * hidden_size + offs_k[:, None]
        )
        part_g = tl.dot(aq, w_gate)
        ws_gate = tl.load(
            w1_scale_ptr + expert * w1s_stride_e + pid_n * w1s_stride_n + kb
        )
        acc_g += part_g * (a_scale[:, None] * ws_gate)

    # Mirror the reference dtype flow: bf16 round -> silu on the rounded value
    # (fp32 math) -> bf16 round.  The stored bf16 silu(g) tile is exactly the
    # factor the fused epilogue multiplied into u, so the split keeps the
    # downstream numerics identical.  It lands at the tile's sorted slot
    # positions (contiguous rows), matching the up pass's contiguous load.
    out_ty = tl.bfloat16
    g = acc_g.to(out_ty).to(tl.float32)
    s = g / (1.0 + tl.exp(-g))
    s = s.to(out_ty)
    tl.store(
        silu_g_ptr + slots[:, None] * inter_size + offs_n[None, :],
        s,
        mask=valid[:, None],
    )


@triton.jit
def _gemm_up_requant_kernel(
    sorted_ptr,
    ids_ptr,
    act_q_ptr,
    act_scale_ptr,
    w1_ptr,
    w1_scale_ptr,
    silu_g_ptr,
    inter_q_ptr,
    inter_scale_ptr,
    npost_ptr,
    num_pairs,
    hidden_size,
    inter_size,
    w1s_stride_e,
    w1s_stride_n,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GN: tl.constexpr,
    GK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BM
    if row_start >= tl.load(npost_ptr):
        return

    first_pair = tl.load(sorted_ptr + row_start)
    if first_pair >= num_pairs:
        return
    expert = tl.load(ids_ptr + first_pair)

    slots = row_start + tl.arange(0, BM)
    pair_ids = tl.load(sorted_ptr + slots)
    valid = pair_ids < num_pairs
    token = pair_ids // TOPK

    offs_n = pid_n * BN + tl.arange(0, BN)
    num_k_blocks = hidden_size // BK
    num_groups = inter_size // GK
    num_act_groups = hidden_size // GK
    up_n_block = inter_size // GN

    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    w1_base = w1_ptr + expert * (2 * inter_size) * hidden_size
    for kb in range(num_k_blocks):
        offs_k = kb * BK + tl.arange(0, BK)
        aq = tl.load(
            act_q_ptr + token[:, None] * hidden_size + offs_k[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        a_scale = tl.load(
            act_scale_ptr + token * num_act_groups + kb,
            mask=valid,
            other=0.0,
        )
        w_up = tl.load(
            w1_base + (offs_n[None, :] + inter_size) * hidden_size + offs_k[:, None]
        )
        part_u = tl.dot(aq, w_up)
        ws_up = tl.load(
            w1_scale_ptr
            + expert * w1s_stride_e
            + (pid_n + up_n_block) * w1s_stride_n
            + kb
        )
        acc_u += part_u * (a_scale[:, None] * ws_up)

    # Same dtype flow as the fused epilogue: bf16 silu(g) from the gate pass,
    # bf16-rounded u, bf16 product, then per-row requantization over this full
    # 128-column group (BN == group_k keeps one whole group inside the tile).
    out_ty = tl.bfloat16
    s = tl.load(
        silu_g_ptr + slots[:, None] * inter_size + offs_n[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    u = acc_u.to(out_ty).to(tl.float32)
    inter = (s * u).to(out_ty).to(tl.float32)

    amax_i = tl.max(tl.abs(inter), axis=1)
    i_scale = tl.maximum(amax_i, SCALE_EPS) / FP8_E4M3_MAX
    iq = inter / i_scale[:, None]
    iq = tl.minimum(tl.maximum(iq, -FP8_E4M3_MAX), FP8_E4M3_MAX)
    iq = iq.to(tl.float8e4nv)

    # Store at the tile's sorted slot positions (contiguous rows) so the down
    # GEMM's activation operand loads are fully coalesced tile reads.
    tl.store(
        inter_q_ptr + slots[:, None] * inter_size + offs_n[None, :],
        iq,
        mask=valid[:, None],
    )
    group_idx = (pid_n * BN) // GK
    tl.store(
        inter_scale_ptr + slots * num_groups + group_idx,
        i_scale,
        mask=valid,
    )


@triton.jit
def _gemm_down_kernel(
    sorted_ptr,
    ids_ptr,
    inter_q_ptr,
    inter_scale_ptr,
    w2_ptr,
    w2_scale_ptr,
    weights_ptr,
    out_f32_ptr,
    npost_ptr,
    num_pairs,
    hidden_size,
    inter_size,
    w2s_stride_e,
    w2s_stride_n,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BM
    if row_start >= tl.load(npost_ptr):
        return

    # Segments pad to BM rows, so every BM-tile holds at least one real pair;
    # the sentinel check is retained defensively: since each segment fills
    # contiguously from its start, a tile whose first entry is the num_pairs
    # sentinel is all padding -> skip it (no weight streaming).
    first_pair = tl.load(sorted_ptr + row_start)
    if first_pair >= num_pairs:
        return
    expert = tl.load(ids_ptr + first_pair)

    slots = row_start + tl.arange(0, BM)
    pair_ids = tl.load(sorted_ptr + slots)
    valid = pair_ids < num_pairs

    offs_n = pid_n * BN + tl.arange(0, BN)
    num_k_blocks = inter_size // BK
    num_groups = inter_size // GK

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    w2_base = w2_ptr + expert * hidden_size * inter_size
    for kb in range(num_k_blocks):
        offs_k = kb * BK + tl.arange(0, BK)
        # Activation operand rows live at the tile's sorted slot positions
        # (contiguous), not at scattered pair ids: coalesced tile reads.
        aq = tl.load(
            inter_q_ptr + slots[:, None] * inter_size + offs_k[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        a_scale = tl.load(
            inter_scale_ptr + slots * num_groups + kb,
            mask=valid,
            other=0.0,
        )
        w = tl.load(w2_base + offs_n[None, :] * inter_size + offs_k[:, None])
        part = tl.dot(aq, w)
        ws = tl.load(
            w2_scale_ptr + expert * w2s_stride_e + pid_n * w2s_stride_n + kb
        )
        acc += part * (a_scale[:, None] * ws)

    # Fused top-k epilogue.  Mirror the reference dtype flow exactly: round
    # the accumulator to bf16 (the old partial-buffer element type), promote
    # back to fp32, multiply by the per-(token, slot) top-k weight, and
    # scatter-add into the fp32 output accumulator.  The flat pair index is
    # token * TOPK + slot, so it indexes topk_weights directly.
    w_topk = tl.load(weights_ptr + pair_ids, mask=valid, other=0.0)
    contrib = acc.to(tl.bfloat16).to(tl.float32) * w_topk[:, None]
    token = pair_ids // TOPK
    tl.atomic_add(
        out_f32_ptr + token[:, None] * hidden_size + offs_n[None, :],
        contrib,
        mask=valid[:, None],
        sem="relaxed",
    )


@triton.jit
def _fp32_cast_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(
        self,
        num_experts: int = 256,
        intermediate_size: int = 512,
        top_k: int = 8,
        block_shape: list[int] | tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.block_shape = tuple(int(value) for value in block_shape)

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
        hidden_states = hidden_states.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.contiguous()
        w1_scale = w1_scale.contiguous()
        w2_scale = w2_scale.contiguous()

        token_count, hidden_size = hidden_states.shape
        experts = self.num_experts
        inter_size = self.intermediate_size
        topk = self.top_k
        group_n, group_k = self.block_shape
        if group_n != BLOCK_N or group_k != BLOCK_K:
            raise ValueError(
                f"block_shape {self.block_shape} does not match the kernel "
                f"tile constants ({BLOCK_N}, {BLOCK_K})"
            )
        if hidden_size % group_k or inter_size % group_k or inter_size % BLOCK_N:
            raise ValueError(
                "hidden/intermediate sizes must be multiples of the block shape"
            )
        num_pairs = token_count * topk
        device = hidden_states.device
        out_dtype = hidden_states.dtype

        output = torch.empty(
            token_count, hidden_size, device=device, dtype=out_dtype
        )
        if num_pairs == 0:
            return output

        # --- dispatch: group (token, expert-slot) pairs by expert ------------
        counts = torch.zeros(experts, device=device, dtype=torch.int32)
        starts = torch.empty(experts, device=device, dtype=torch.int32)
        cursor = torch.zeros(experts, device=device, dtype=torch.int32)
        npost = torch.empty(1, device=device, dtype=torch.int32)

        # Dispatch padding: expert segments pad to BLOCK_M rows on every
        # workload.  That is the minimum padding compatible with single-expert
        # 64-row GEMM tiles; the former dense-workload 128-row padding existed
        # for a 128-row down GEMM and became pure waste (~63 padded rows per
        # expert on average) once the down GEMM moved to BLOCK_M tiles.
        pad_bm = BLOCK_M

        num_blocks_max = (num_pairs + experts * (pad_bm - 1) + pad_bm - 1) // pad_bm
        sorted_ids = torch.full(
            (num_blocks_max * pad_bm,), num_pairs, device=device, dtype=torch.int32
        )

        dispatch_grid = (triton.cdiv(num_pairs, DISPATCH_BLOCK),)
        _expert_count_kernel[dispatch_grid](
            topk_ids, counts, num_pairs, BLOCK=DISPATCH_BLOCK, num_warps=4
        )
        _expert_scan_kernel[(1,)](
            counts,
            starts,
            npost,
            experts,
            BM=pad_bm,
            BLOCK_E=triton.next_power_of_2(experts),
            num_warps=4,
        )
        _token_scatter_kernel[dispatch_grid](
            topk_ids, starts, cursor, sorted_ids, num_pairs,
            BLOCK=DISPATCH_BLOCK, num_warps=4,
        )

        # --- activation pre-quantization (once per token) --------------------
        # Bit-identical to the former in-loop quantization; the gate-up GEMM
        # consumes the fp8 values and per-group scales instead of requantizing
        # every routed slot copy of every token row.
        act_num_groups = hidden_size // group_k
        act_q = torch.empty(
            token_count, hidden_size, device=device, dtype=torch.float8_e4m3fn
        )
        act_scale = torch.empty(
            token_count, act_num_groups, device=device, dtype=torch.float32
        )
        total_act_groups = token_count * act_num_groups
        _activation_quant_kernel[(triton.cdiv(total_act_groups, ACT_GROUPS),)](
            hidden_states,
            act_q,
            act_scale,
            total_act_groups,
            GROUPS=ACT_GROUPS,
            num_warps=4,
        )

        # --- GEMM1 + SiLU*up + requantize (split single-accumulator passes) --
        # The inter-pass buffers are indexed by sorted slot position (the tile
        # row index inside the padded expert segments) rather than by pair id:
        # every tile's 64 rows are contiguous in these buffers, so the gate/up
        # epilogue stores and the down GEMM's activation operand loads are
        # fully coalesced tile I/O instead of 64-row scatters/gathers across
        # pair-major memory.  The buffers span the same extent as sorted_ids;
        # positions beyond npost are never written or read (the GEMM grids
        # early-return past npost and padding rows stay valid-masked).
        inter_rows = num_blocks_max * pad_bm
        inter_q = torch.empty(
            inter_rows, inter_size, device=device, dtype=torch.float8_e4m3fn
        )
        inter_scale = torch.empty(
            inter_rows, inter_size // group_k, device=device, dtype=torch.float32
        )
        silu_g = torch.empty(
            inter_rows, inter_size, device=device, dtype=torch.bfloat16
        )
        w1s_stride_e = ((2 * inter_size) // group_n) * (hidden_size // group_k)
        w1s_stride_n = hidden_size // group_k
        gate_up_grid = (
            num_blocks_max * (pad_bm // BLOCK_M),
            inter_size // BLOCK_N,
        )
        # Two CTAs per SM need <=128 registers per thread of the 65536-entry
        # register file; maxnreg enforces that budget (the single-accumulator
        # dataflow fits it, the old dual-dot did not).  num_stages=3 is the
        # deepest pipeline whose doubled smem still leaves room for two CTAs.
        _gemm_gate_kernel[gate_up_grid](
            sorted_ids,
            topk_ids,
            act_q,
            act_scale,
            w1,
            w1_scale,
            silu_g,
            npost,
            num_pairs,
            hidden_size,
            inter_size,
            w1s_stride_e,
            w1s_stride_n,
            TOPK=topk,
            BM=BLOCK_M,
            BN=BLOCK_N,
            BK=BLOCK_K,
            GK=group_k,
            num_warps=8,
            num_stages=3,
            maxnreg=128,
        )
        _gemm_up_requant_kernel[gate_up_grid](
            sorted_ids,
            topk_ids,
            act_q,
            act_scale,
            w1,
            w1_scale,
            silu_g,
            inter_q,
            inter_scale,
            npost,
            num_pairs,
            hidden_size,
            inter_size,
            w1s_stride_e,
            w1s_stride_n,
            TOPK=topk,
            BM=BLOCK_M,
            BN=BLOCK_N,
            BK=BLOCK_K,
            GN=group_n,
            GK=group_k,
            num_warps=8,
            num_stages=3,
            maxnreg=128,
        )

        # --- GEMM2 with fused top-k weighting -------------------------------
        out_f32 = torch.zeros(
            token_count, hidden_size, device=device, dtype=torch.float32
        )
        w2s_stride_e = (hidden_size // group_n) * (inter_size // group_k)
        w2s_stride_n = inter_size // group_k
        # The down GEMM tiles at BLOCK_M rows on every workload: the 64x128
        # fp32 accumulator is the 32-regs/thread class that fits the two-CTA
        # budget (<=128 regs/thread with maxnreg=128, stages=3 smem x2 CTAs
        # <= 128KB), whereas 128-row tiles need a 64-regs/thread accumulator
        # that spills under the cap.  The former uncapped 4-stage variant
        # compiled at 172 registers (73728B smem) and resided one CTA/SM,
        # halving the resident warps of the down phase.
        down_grid = (
            num_blocks_max * (pad_bm // BLOCK_M),
            hidden_size // BLOCK_N,
        )
        down_caps = {"maxnreg": 128}
        _gemm_down_kernel[down_grid](
            sorted_ids,
            topk_ids,
            inter_q,
            inter_scale,
            w2,
            w2_scale,
            topk_weights,
            out_f32,
            npost,
            num_pairs,
            hidden_size,
            inter_size,
            w2s_stride_e,
            w2s_stride_n,
            TOPK=topk,
            BM=BLOCK_M,
            BN=BLOCK_N,
            BK=BLOCK_K,
            GK=group_k,
            num_warps=8,
            num_stages=3,
            **down_caps,
        )

        # --- cast the accumulated fp32 output to the activation dtype --------
        n_elements = token_count * hidden_size
        cast_block = 1024
        _fp32_cast_kernel[(triton.cdiv(n_elements, cast_block),)](
            out_f32,
            output,
            n_elements,
            BLOCK=cast_block,
            num_warps=4,
        )
        return output
