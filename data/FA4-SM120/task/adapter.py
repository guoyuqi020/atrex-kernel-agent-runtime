"""Fixed production ABI adapter for a self-authored SM120 CuTe implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from implementation.sm120 import flash_attention_sm120


class Model(nn.Module):
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
        fp8 = torch.float8_e4m3fn
        if query.ndim != 3 or tuple(query.shape[1:]) != (16, 256):
            raise ValueError("query must have shape [total_q, 16, 256]")
        if kv_cache.ndim != 5 or tuple(kv_cache.shape[1:]) != (2, 1, 64, 256):
            raise ValueError("kv_cache must have shape [pages, 2, 1, 64, 256]")
        if query.dtype != fp8 or kv_cache.dtype != fp8:
            raise ValueError("query and kv_cache must use FP8 e4m3fn")
        if out.dtype != torch.bfloat16 or tuple(out.shape) != tuple(query.shape):
            raise ValueError("out must be BF16 with the same shape as query")
        if block_tables.ndim != 2 or block_tables.dtype != torch.int32:
            raise ValueError("block_tables must be rank-2 int32")
        if seq_lens.ndim != 1 or seq_lens.dtype != torch.int32:
            raise ValueError("seq_lens must be rank-1 int32")
        if cum_seq_lens_q.ndim != 1 or cum_seq_lens_q.dtype != torch.int32:
            raise ValueError("cum_seq_lens_q must be rank-1 int32")
        if cum_seq_lens_kv.ndim != 1 or cum_seq_lens_kv.dtype != torch.int32:
            raise ValueError("cum_seq_lens_kv must be rank-1 int32")
        if batch_size != seq_lens.numel() or block_tables.shape[0] != batch_size:
            raise ValueError("batch metadata dimensions disagree")
        if cum_seq_lens_q.numel() != batch_size + 1:
            raise ValueError("cum_seq_lens_q must contain batch_size + 1 entries")
        if cum_seq_lens_kv.numel() != batch_size + 1:
            raise ValueError("cum_seq_lens_kv must contain batch_size + 1 entries")
        if bmm2_scale != 1.0:
            raise ValueError("target contract requires bmm2_scale == 1")
        if window_left != -1 or sinks is not None or o_sf_scale is not None:
            raise ValueError("target is full causal attention without sinks/output scaling")

        result = flash_attention_sm120(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=batch_size,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            window_left=window_left,
            sinks=sinks,
            o_sf_scale=o_sf_scale,
            out=out,
        )
        if result is not out:
            raise RuntimeError(
                "SM120 implementation must mutate and return the supplied out tensor"
            )
        return out
