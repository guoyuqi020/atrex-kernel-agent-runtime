from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        window_left: int,
        sinks: torch.Tensor | None,
        o_sf_scale: torch.Tensor | None,
        out: torch.Tensor,
    ) -> torch.Tensor:
        del workspace_buffer, max_q_len, max_kv_len
        if window_left != -1 or sinks is not None or o_sf_scale is not None:
            raise ValueError("captured Qwen3.8-Max path uses full causal attention")
        if batch_size != seq_lens.numel():
            raise ValueError("batch_size and seq_lens disagree")
        if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
            raise ValueError("expected HND KV cache")

        page_size = kv_cache.shape[-2]
        num_q_heads = query.shape[1]
        num_kv_heads = kv_cache.shape[2]
        if num_q_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        q_heads_per_kv = num_q_heads // num_kv_heads

        query_boundaries = cum_seq_lens_q.tolist()
        page_boundaries = cum_seq_lens_kv.tolist()
        sequence_lengths = seq_lens.tolist()
        for request_index, seq_len in enumerate(sequence_lengths):
            query_start = query_boundaries[request_index]
            query_end = query_boundaries[request_index + 1]
            query_len = query_end - query_start
            page_count = page_boundaries[request_index + 1] - page_boundaries[
                request_index
            ]
            expected_pages = (seq_len + page_size - 1) // page_size
            if page_count != expected_pages:
                raise ValueError("page indptr does not match sequence length")

            page_ids = block_tables[request_index, :page_count].to(
                dtype=torch.long
            )
            pages = kv_cache.index_select(0, page_ids)
            keys = (
                pages[:, 0]
                .permute(0, 2, 1, 3)
                .reshape(page_count * page_size, num_kv_heads, -1)[:seq_len]
            )
            values = (
                pages[:, 1]
                .permute(0, 2, 1, 3)
                .reshape(page_count * page_size, num_kv_heads, -1)[:seq_len]
            )

            query_position_start = seq_len - query_len
            for kv_head in range(num_kv_heads):
                head_start = kv_head * q_heads_per_kv
                head_end = head_start + q_heads_per_kv
                key = keys[:, kv_head].float()
                value = values[:, kv_head].float()
                for chunk_start in range(0, query_len, 64):
                    chunk_end = min(query_len, chunk_start + 64)
                    q_chunk = query[
                        query_start + chunk_start : query_start + chunk_end,
                        head_start:head_end,
                    ].float()
                    scores = (
                        torch.einsum("qhd,kd->hqk", q_chunk, key)
                        * bmm1_scale
                    )
                    query_positions = (
                        torch.arange(
                            chunk_start,
                            chunk_end,
                            device=query.device,
                        )
                        + query_position_start
                    )
                    key_positions = torch.arange(
                        seq_len, device=query.device
                    )
                    causal = key_positions.unsqueeze(0) <= (
                        query_positions.unsqueeze(1)
                    )
                    scores.masked_fill_(
                        ~causal.unsqueeze(0), float("-inf")
                    )
                    probabilities = torch.softmax(scores, dim=-1)
                    output = (
                        torch.einsum(
                            "hqk,kd->qhd", probabilities, value
                        )
                        * bmm2_scale
                    )
                    out[
                        query_start + chunk_start : query_start + chunk_end,
                        head_start:head_end,
                    ].copy_(output.to(dtype=out.dtype))
        return out
