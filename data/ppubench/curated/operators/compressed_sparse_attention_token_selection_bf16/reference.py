"""Source-bound reference for the exact Qwen3.8 CSA token-selection call.

Derived from qwen_compressed_sparse_attn_indexer.py at source commit
a9ffcd92cb47750d8f8ddd0663e0843fc84ca2b2. The helper bodies are copied
without algorithm changes; wrapper only adapts the benchmark Model ABI.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


CSA_PREFILL_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024


def _validate_mqa_inputs(q: torch.Tensor, k: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(
            f"CSA requires q [tokens, heads, head_dim], got {tuple(q.shape)}"
        )
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(
            f"CSA MQA requires k [tokens, 1, head_dim], got {tuple(k.shape)}"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("CSA query and key head dimensions must match")
    if q.device != k.device:
        raise ValueError("CSA query and key tensors must be on the same device")


def csa_weight_free_mqa_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Compute ``sum_heads(relu(q @ k)) / scale`` in FP32."""

    _validate_mqa_inputs(q, k)
    scale = math.sqrt(q.shape[-1]) if score_scale is None else score_scale
    if scale <= 0:
        raise ValueError("score_scale must be positive")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    return torch.relu(scores).sum(dim=-1) / scale


def _validate_row_ranges(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if row_starts.ndim != 1 or row_ends.ndim != 1:
        raise ValueError("CSA row starts and ends must be one-dimensional")
    if row_starts.numel() != rows or row_ends.numel() != rows:
        raise ValueError("CSA row ranges must have one entry per query")
    starts = row_starts.to(dtype=torch.long)
    ends = row_ends.to(dtype=torch.long)
    if rows and (
        torch.any(starts < 0) or torch.any(starts > ends) or torch.any(ends > columns)
    ):
        raise ValueError("CSA row ranges must satisfy 0 <= start <= end <= keys")
    return starts, ends


def csa_mqa_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Score packed variable-length prefill rows without crossing sequences."""

    logits = csa_weight_free_mqa_scores(q, k, score_scale)
    starts, ends = _validate_row_ranges(row_starts, row_ends, q.shape[0], k.shape[0])
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= starts.to(q.device).unsqueeze(1)) & (
        columns < ends.to(q.device).unsqueeze(1)
    )
    return logits.masked_fill(~valid, -torch.inf)


def csa_relative_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select top compressed blocks and return row-relative indices."""

    if logits.ndim != 2:
        raise ValueError("CSA logits must be a two-dimensional tensor")
    if topk <= 0:
        raise ValueError("topk must be positive")
    starts, ends = _validate_row_ranges(
        row_starts, row_ends, logits.shape[0], logits.shape[1]
    )
    output = torch.full(
        (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
    )
    for row in range(logits.shape[0]):
        start = int(starts[row])
        length = int(ends[row] - starts[row])
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def expand_csa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed blocks and append the visible incomplete tail."""

    if compress_ratio <= 0 or token_topk <= 0:
        raise ValueError("compress_ratio and token_topk must be positive")
    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [rows, {block_topk}], "
            f"got {tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")

    device = block_indices.device
    blocks = block_indices.long()
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]

    query_positions = query_positions.to(device=device, dtype=torch.long)
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible_tokens = query_positions + 1
    tail_start = (
        torch.div(visible_tokens, compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(final_topk, device=device).unsqueeze(0).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + final_topk)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


def csa_prefill_row_chunk_size(
    rows: int,
    keys: int,
    heads: int,
    logits_budget_bytes: int = CSA_PREFILL_LOGITS_BUDGET_BYTES,
) -> int:
    """Choose a row tile that bounds the padded FP32 logits workspace."""

    if rows < 0 or keys < 0 or heads <= 0:
        raise ValueError(
            "rows and keys must be non-negative and heads must be positive"
        )
    if logits_budget_bytes <= 0:
        raise ValueError("logits_budget_bytes must be positive")
    if rows == 0 or keys == 0:
        return max(rows, 1)

    block_q = max(1, 128 // heads)
    bytes_per_row = keys * torch.float32.itemsize
    max_padded_rows = max(block_q, logits_budget_bytes // bytes_per_row)
    max_padded_rows = max(block_q, max_padded_rows // block_q * block_q)
    return min(rows, max_padded_rows)


def select_csa_prefill_tokens_reference(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    logits_budget_bytes: int = CSA_PREFILL_LOGITS_BUDGET_BYTES,
) -> torch.Tensor:
    """Select prefill tokens while bounding the temporary FP32 logits matrix."""

    if token_topk <= 0 or compress_ratio <= 0:
        raise ValueError("token_topk and compress_ratio must be positive")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    _validate_mqa_inputs(q, compressed_keys)
    starts, ends = _validate_row_ranges(
        row_starts, row_ends, q.shape[0], compressed_keys.shape[0]
    )
    if query_positions.numel() != q.shape[0]:
        raise ValueError("query positions must have one entry per query")
    if sequence_lengths.numel() != q.shape[0]:
        raise ValueError("sequence lengths must have one entry per query")

    rows = q.shape[0]
    output = torch.empty(
        (rows, token_topk + compress_ratio - 1),
        dtype=torch.int32,
        device=q.device,
    )
    if rows == 0:
        return output

    block_topk = token_topk // compress_ratio
    row_chunk_size = csa_prefill_row_chunk_size(
        rows,
        compressed_keys.shape[0],
        q.shape[1],
        logits_budget_bytes,
    )
    for row_start in range(0, rows, row_chunk_size):
        row_end = min(row_start + row_chunk_size, rows)
        row_slice = slice(row_start, row_end)
        if compressed_keys.shape[0] == 0:
            block_indices = torch.full(
                (row_end - row_start, block_topk),
                -1,
                dtype=torch.int32,
                device=q.device,
            )
        else:
            logits = csa_mqa_prefill_reference(
                q[row_slice],
                compressed_keys,
                starts[row_slice],
                ends[row_slice],
            )
            block_indices = csa_relative_topk(
                logits, starts[row_slice], ends[row_slice], block_topk
            )
        output[row_slice].copy_(
            expand_csa_block_indices(
                block_indices,
                query_positions[row_slice],
                sequence_lengths[row_slice],
                compress_ratio,
                token_topk,
            )
        )
    return output


class Model(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        compressed_keys: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        token_topk: int,
        compress_ratio: int,
    ) -> torch.Tensor:
        return select_csa_prefill_tokens_reference(
            q, compressed_keys, row_starts, row_ends, query_positions,
            sequence_lengths, token_topk, compress_ratio
        )
