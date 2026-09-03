"""Block-scaled FP8 fused MoE implemented entirely in self-authored Triton kernels.

Every compute step of the reference operator is expressed as a Triton kernel:

1. ``_expert_count_kernel``  - histogram of flattened top-k expert ids.
2. ``_expert_scan_kernel``   - exclusive scan of per-expert token counts padded to
                               ``BLOCK_M`` so every GEMM tile belongs to one expert.
3. ``_token_scatter_kernel`` - gathers the (token, expert-slot) pairs into the
                               padded, expert-major ``sorted_token_ids`` layout.
4. ``_prequant_kernel``      - one-time per-128-K-group FP8 quantization of
                               ``hidden_states`` (scale = max(amax, 1e-12)/448,
                               RTNE cast), exactly the reference
                               ``_quantize_activation`` math, so the gate/up
                               GEMM consumes FP8 activations instead of
                               requantizing the same rows per tile.
5. ``_gemm_gate_up_kernel``  - grouped GEMM against ``w1`` over the
                               prequantized FP8 activations/scales; the
                               block-scaled matmul accumulates in FP32, and the
                               epilogue fuses the bf16 rounding, SiLU-gate * up
                               product, and requantization of the intermediate.
6. ``_gemm_down_kernel``     - grouped GEMM against ``w2`` with the stored
                               intermediate FP8 values/scales, FP32 accumulate.
7. ``_topk_reduce_kernel``   - FP32 weighted sum over the eight routed expert
                               outputs, cast back to the activation dtype.

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

BLOCK_M = 64  # rows per grouped-GEMM tile; expert segments are padded to this
BLOCK_N = 128  # matches block_shape[0]; keeps weight-scale blocks tile-aligned
BLOCK_K = 128  # matches block_shape[1]; activation quantization group width
DISPATCH_BLOCK = 1024  # lanes per routing histogram/scatter program


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
def _prequant_kernel(
    hidden_ptr,
    q_ptr,
    scale_ptr,
    hidden_size,
    num_tokens,
    TQ: tl.constexpr,
    GK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * TQ + tl.arange(0, TQ)
    row_mask = rows < num_tokens
    num_groups = hidden_size // GK
    offs_gk = tl.arange(0, GK)
    for g in range(num_groups):
        offs = rows[:, None] * hidden_size + g * GK + offs_gk[None, :]
        m = row_mask[:, None]
        x = tl.load(hidden_ptr + offs, mask=m, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        scale = tl.maximum(amax, SCALE_EPS) / FP8_E4M3_MAX
        q = x / scale[:, None]
        q = tl.minimum(tl.maximum(q, -FP8_E4M3_MAX), FP8_E4M3_MAX)
        tl.store(q_ptr + offs, q.to(tl.float8e4nv), mask=m)
        tl.store(scale_ptr + rows * num_groups + g, scale, mask=row_mask)


@triton.jit
def _gemm_gate_up_kernel(
    sorted_ptr,
    ids_ptr,
    hidden_ptr,
    hidden_q_ptr,
    hidden_scale_ptr,
    w1_ptr,
    w1_scale_ptr,
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

    # Every tile inside the padded region starts on a real (token, slot) pair,
    # so the first entry identifies the tile's expert.
    first_pair = tl.load(sorted_ptr + row_start)
    expert = tl.load(ids_ptr + first_pair)

    slots = row_start + tl.arange(0, BM)
    pair_ids = tl.load(sorted_ptr + slots)
    valid = pair_ids < num_pairs
    token = pair_ids // TOPK

    offs_n = pid_n * BN + tl.arange(0, BN)
    num_k_blocks = hidden_size // BK
    num_groups = inter_size // GK
    up_n_block = inter_size // GN

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    w1_base = w1_ptr + expert * (2 * inter_size) * hidden_size
    hnum_groups = hidden_size // GK
    for kb in range(num_k_blocks):
        offs_k = kb * BK + tl.arange(0, BK)
        # Prequantized FP8 activations: one quantization per (token, 128-K
        # group) done once in _prequant_kernel instead of per GEMM tile.
        aq = tl.load(
            hidden_q_ptr + token[:, None] * hidden_size + offs_k[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        a_scale = tl.load(
            hidden_scale_ptr + token * hnum_groups + kb,
            mask=valid,
            other=0.0,
        )

        w_gate = tl.load(
            w1_base + offs_n[None, :] * hidden_size + offs_k[:, None]
        )
        w_up = tl.load(
            w1_base + (offs_n[None, :] + inter_size) * hidden_size + offs_k[:, None]
        )
        part_g = tl.dot(aq, w_gate)
        part_u = tl.dot(aq, w_up)

        ws_gate = tl.load(
            w1_scale_ptr + expert * w1s_stride_e + pid_n * w1s_stride_n + kb
        )
        ws_up = tl.load(
            w1_scale_ptr
            + expert * w1s_stride_e
            + (pid_n + up_n_block) * w1s_stride_n
            + kb
        )
        acc_g += part_g * (a_scale[:, None] * ws_gate)
        acc_u += part_u * (a_scale[:, None] * ws_up)

    # Mirror the reference dtype flow: bf16 round -> silu on the rounded value
    # (fp32 math) -> bf16 round -> bf16 product -> quantize from the bf16 value.
    out_ty = hidden_ptr.dtype.element_ty
    g = acc_g.to(out_ty).to(tl.float32)
    u = acc_u.to(out_ty).to(tl.float32)
    s = g / (1.0 + tl.exp(-g))
    s = s.to(out_ty).to(tl.float32)
    inter = (s * u).to(out_ty).to(tl.float32)

    amax_i = tl.max(tl.abs(inter), axis=1)
    i_scale = tl.maximum(amax_i, SCALE_EPS) / FP8_E4M3_MAX
    iq = inter / i_scale[:, None]
    iq = tl.minimum(tl.maximum(iq, -FP8_E4M3_MAX), FP8_E4M3_MAX)
    iq = iq.to(tl.float8e4nv)

    tl.store(
        inter_q_ptr + pair_ids[:, None] * inter_size + offs_n[None, :],
        iq,
        mask=valid[:, None],
    )
    group_idx = (pid_n * BN) // GK
    tl.store(
        inter_scale_ptr + pair_ids * num_groups + group_idx,
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
    out_ptr,
    npost_ptr,
    num_pairs,
    hidden_size,
    inter_size,
    w2s_stride_e,
    w2s_stride_n,
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

    first_pair = tl.load(sorted_ptr + row_start)
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
        aq = tl.load(
            inter_q_ptr + pair_ids[:, None] * inter_size + offs_k[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        a_scale = tl.load(
            inter_scale_ptr + pair_ids * num_groups + kb,
            mask=valid,
            other=0.0,
        )
        w = tl.load(w2_base + offs_n[None, :] * inter_size + offs_k[:, None])
        part = tl.dot(aq, w)
        ws = tl.load(
            w2_scale_ptr + expert * w2s_stride_e + pid_n * w2s_stride_n + kb
        )
        acc += part * (a_scale[:, None] * ws)

    tl.store(
        out_ptr + pair_ids[:, None] * hidden_size + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=valid[:, None],
    )


@triton.jit
def _topk_reduce_kernel(
    part_ptr,
    weights_ptr,
    out_ptr,
    num_tokens,
    hidden_size,
    TOPK: tl.constexpr,
    BMT: tl.constexpr,
    BNH: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BMT + tl.arange(0, BMT)
    offs_h = pid_h * BNH + tl.arange(0, BNH)
    mask_t = offs_t < num_tokens

    acc = tl.zeros((BMT, BNH), dtype=tl.float32)
    for k in range(TOPK):
        rows = offs_t * TOPK + k
        part = tl.load(
            part_ptr + rows[:, None] * hidden_size + offs_h[None, :],
            mask=mask_t[:, None],
            other=0.0,
        )
        w = tl.load(weights_ptr + offs_t * TOPK + k, mask=mask_t, other=0.0)
        acc += part.to(tl.float32) * w[:, None]

    tl.store(
        out_ptr + offs_t[:, None] * hidden_size + offs_h[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_t[:, None],
    )


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

        num_blocks_max = (num_pairs + experts * (BLOCK_M - 1) + BLOCK_M - 1) // BLOCK_M
        sorted_ids = torch.full(
            (num_blocks_max * BLOCK_M,), num_pairs, device=device, dtype=torch.int32
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
            BM=BLOCK_M,
            BLOCK_E=triton.next_power_of_2(experts),
            num_warps=4,
        )
        _token_scatter_kernel[dispatch_grid](
            topk_ids, starts, cursor, sorted_ids, num_pairs,
            BLOCK=DISPATCH_BLOCK, num_warps=4,
        )

        # --- one-time activation quantization (per 128-K group) --------------
        hidden_q = torch.empty(
            token_count, hidden_size, device=device, dtype=torch.float8_e4m3fn
        )
        hidden_scale = torch.empty(
            token_count, hidden_size // group_k, device=device, dtype=torch.float32
        )
        prequant_tq = 4
        _prequant_kernel[(triton.cdiv(token_count, prequant_tq),)](
            hidden_states,
            hidden_q,
            hidden_scale,
            hidden_size,
            token_count,
            TQ=prequant_tq,
            GK=group_k,
            num_warps=4,
        )

        # --- GEMM1 + SiLU*up + requantize ------------------------------------
        inter_q = torch.empty(
            num_pairs, inter_size, device=device, dtype=torch.float8_e4m3fn
        )
        inter_scale = torch.empty(
            num_pairs, inter_size // group_k, device=device, dtype=torch.float32
        )
        w1s_stride_e = ((2 * inter_size) // group_n) * (hidden_size // group_k)
        w1s_stride_n = hidden_size // group_k
        _gemm_gate_up_kernel[(num_blocks_max, inter_size // BLOCK_N)](
            sorted_ids,
            topk_ids,
            hidden_states,
            hidden_q,
            hidden_scale,
            w1,
            w1_scale,
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
            num_warps=16,
            num_stages=1,
        )

        # --- GEMM2 -------------------------------------------------------------
        partial = torch.empty(num_pairs, hidden_size, device=device, dtype=out_dtype)
        w2s_stride_e = (hidden_size // group_n) * (inter_size // group_k)
        w2s_stride_n = inter_size // group_k
        _gemm_down_kernel[(num_blocks_max, hidden_size // BLOCK_N)](
            sorted_ids,
            topk_ids,
            inter_q,
            inter_scale,
            w2,
            w2_scale,
            partial,
            npost,
            num_pairs,
            hidden_size,
            inter_size,
            w2s_stride_e,
            w2s_stride_n,
            BM=BLOCK_M,
            BN=BLOCK_N,
            BK=BLOCK_K,
            GK=group_k,
            num_warps=4,
            num_stages=3,
        )

        # --- top-k weighted reduce ---------------------------------------------
        reduce_bmt = 64
        reduce_bnh = 128
        _topk_reduce_kernel[
            (triton.cdiv(token_count, reduce_bmt), triton.cdiv(hidden_size, reduce_bnh))
        ](
            partial,
            topk_weights,
            output,
            token_count,
            hidden_size,
            TOPK=topk,
            BMT=reduce_bmt,
            BNH=reduce_bnh,
            num_warps=4,
        )
        return output
