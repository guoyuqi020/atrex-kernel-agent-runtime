"""Source-bound reference for Qwen3.8 CSA paged BF16 attention.

Derived from qwen4_exp_csa.py and qwen_compressed_sparse_attn_indexer.py
at source commit a9ffcd92cb47750d8f8ddd0663e0843fc84ca2b2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CSAProductionKernelUnavailable(RuntimeError):
    pass


def logical_to_physical_csa_slots(
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_block_size: int,
) -> torch.Tensor:
    """Translate request-relative positions into flattened physical slots."""

    if storage_block_size <= 0:
        raise ValueError("CSA storage block size must be positive")
    if block_table.ndim != 2:
        raise ValueError("CSA block table must be two-dimensional")

    requests, positions = torch.broadcast_tensors(request_indices, logical_positions)
    requests = requests.to(device=block_table.device, dtype=torch.long)
    positions = positions.to(device=block_table.device, dtype=torch.long)
    invalid_slots = torch.full_like(positions, -1)
    if block_table.shape[0] == 0 or block_table.shape[1] == 0:
        return invalid_slots

    logical_blocks = torch.div(
        positions.clamp_min(0), storage_block_size, rounding_mode="floor"
    )
    valid = (
        (requests >= 0)
        & (requests < block_table.shape[0])
        & (positions >= 0)
        & (logical_blocks < block_table.shape[1])
    )
    safe_requests = requests.clamp(0, block_table.shape[0] - 1)
    safe_blocks = logical_blocks.clamp(0, block_table.shape[1] - 1)
    physical_blocks = block_table[safe_requests, safe_blocks].long()
    valid &= physical_blocks >= 0
    slots = physical_blocks * storage_block_size + positions.remainder(
        storage_block_size
    )
    return torch.where(valid, slots, invalid_slots)


def csa_logical_to_physical_slots(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map fixed-width request-relative CSA indices to main-cache slots."""

    if logical_indices.ndim != 2:
        raise ValueError("CSA logical indices must be [query_tokens, topk]")
    if token_to_req.ndim != 1 or token_to_req.numel() != logical_indices.shape[0]:
        raise ValueError("CSA token-to-request mapping must match query rows")
    requests = token_to_req.to(device=block_table.device, dtype=torch.long)
    if requests.numel() and (
        torch.any(requests < 0) or torch.any(requests >= sequence_lengths.numel())
    ):
        raise ValueError("CSA token-to-request mapping is out of range")
    row_lengths = sequence_lengths.to(block_table.device).index_select(0, requests)
    logical = logical_indices.to(device=block_table.device, dtype=torch.long)
    valid = (logical >= 0) & (logical < row_lengths.unsqueeze(1))
    slots = logical_to_physical_csa_slots(
        block_table,
        requests.unsqueeze(1),
        logical.clamp_min(0),
        block_size,
    )
    return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)


def csa_sparse_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Compute sparse grouped-query attention over selected physical slots."""

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache and v_cache must be rank-3 tensors")
    if token_slots.ndim != 2 or token_slots.shape[0] != q.shape[0]:
        raise ValueError("token_slots must have one row per query token")
    if k_cache.shape != v_cache.shape:
        raise ValueError("CSA key and value caches must have the same shape")
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("CSA query and cache head dimensions must match")
    if not k_cache.shape[1] or q.shape[1] % k_cache.shape[1]:
        raise ValueError("CSA query heads must be divisible by KV heads")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("CSA attention tensors must be on the same device")

    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    if scale <= 0:
        raise ValueError("softmax_scale must be positive")
    if q.shape[0] == 0:
        return torch.empty_like(q)

    outputs = []
    repeats = q.shape[1] // k_cache.shape[1]
    for row in range(q.shape[0]):
        slots = token_slots[row, token_slots[row] >= 0].long()
        if slots.numel() == 0:
            outputs.append(torch.zeros_like(q[row]))
            continue
        if torch.any(slots >= k_cache.shape[0]):
            raise ValueError("CSA token slots contain an out-of-range cache slot")
        keys = k_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        values = v_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum("hk,khd->hd", probabilities, values.float()).to(q.dtype)
        )
    return torch.stack(outputs)


def csa_sparse_attention_from_logical_indices(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_size: int,
    softmax_scale: float | None = None,
    *,
    allow_cuda_reference: bool = False,
) -> torch.Tensor:
    """Run explicit sparse GQA from request-relative logical indices."""

    if q.is_cuda and not allow_cuda_reference:
        raise CSAProductionKernelUnavailable(
            "Qwen4-Exp CSA requires a fused CUDA sparse-attention kernel; pass "
            "allow_cuda_reference=True only for validation"
        )
    physical_slots = csa_logical_to_physical_slots(
        logical_indices,
        block_table,
        token_to_req,
        sequence_lengths,
        block_size,
    )
    return csa_sparse_attention_reference(
        q,
        key_cache,
        value_cache,
        physical_slots,
        softmax_scale,
    )


class Model(nn.Module):
    def __init__(
        self,
        scale: float,
        kv_cache_dtype: str = "auto",
        layer_k_scale: torch.Tensor | float = 1.0,
        layer_v_scale: torch.Tensor | float = 1.0,
    ) -> None:
        super().__init__()
        if scale <= 0 or kv_cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("CSA scale or cache dtype differs from observed ABI")
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        self.register_buffer("layer_k_scale", torch.as_tensor(layer_k_scale, dtype=torch.float32))
        self.register_buffer("layer_v_scale", torch.as_tensor(layer_v_scale, dtype=torch.float32))
        if self.layer_k_scale.numel() != 1 or self.layer_v_scale.numel() != 1:
            raise ValueError("CSA layer-scale ABI differs")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        topk_indices_buffer: torch.Tensor,
        metadata_present: bool,
        num_actual_tokens: int,
        skip_write_kvcache: bool,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        req_id_per_token: torch.Tensor,
        seq_lens: torch.Tensor,
        num_reqs: int,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("CSA does not support fused output quantization")
        output.zero_()
        if not metadata_present:
            return output
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise ValueError("CSA expects BF16 QKV")
        if kv_cache.dtype != torch.bfloat16 or kv_cache.shape[0] != 2:
            raise ValueError("CSA main cache ABI differs")
        key_cache, value_cache = kv_cache.unbind(0)
        if not skip_write_kvcache:
            slots = slot_mapping[:num_actual_tokens].to(
                device=kv_cache.device, dtype=torch.long
            )
            valid = slots >= 0
            if torch.any(valid):
                valid_slots = slots[valid]
                capacity = kv_cache.shape[1] * kv_cache.shape[2]
                if torch.any(valid_slots >= capacity):
                    raise ValueError("CSA main-cache slot mapping is out of range")
                blocks = torch.div(
                    valid_slots, kv_cache.shape[2], rounding_mode="floor"
                )
                offsets = valid_slots.remainder(kv_cache.shape[2])
                kv_cache[0, blocks, offsets, 0] = key[:num_actual_tokens][valid, 0]
                kv_cache[1, blocks, offsets, 0] = value[:num_actual_tokens][valid, 0]
        indices = topk_indices_buffer[:num_actual_tokens]
        if indices.shape[0] != num_actual_tokens:
            raise ValueError("CSA index buffer is too small")
        reference_output = csa_sparse_attention_from_logical_indices(
            query[:num_actual_tokens],
            key_cache.flatten(0, 1),
            value_cache.flatten(0, 1),
            indices,
            block_table,
            req_id_per_token[:num_actual_tokens],
            seq_lens[:num_reqs],
            key_cache.shape[1],
            self.scale,
            allow_cuda_reference=True,
        )
        output[:num_actual_tokens].copy_(reference_output)
        return output
