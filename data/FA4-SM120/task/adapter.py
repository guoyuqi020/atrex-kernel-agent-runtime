"""C05 production-ABI adapter over the pristine FA4 b54df166 source tree.

This file intentionally contains no C05/Increment kernel optimization.  It
adapts the benchmark ABI to FA4's private forward interface and leaves the
P64, ragged-length, and PackGQA capability gaps visible to the optimizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _path in (ROOT / "vendor_support", ROOT / "vendor/flash_attention"):
    sys.path.insert(0, str(_path))

import torch
import torch.nn as nn

from flash_attn.cute.interface import _flash_attn_fwd


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
        del workspace_buffer, cum_seq_lens_kv
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
        if batch_size != seq_lens.numel() or block_tables.shape[0] != batch_size:
            raise ValueError("batch metadata dimensions disagree")
        if cum_seq_lens_q.numel() != batch_size + 1:
            raise ValueError("cum_seq_lens_q must contain batch_size + 1 entries")
        if bmm2_scale != 1.0:
            raise ValueError("FA4 R0 requires bmm2_scale == 1")
        if window_left != -1 or sinks is not None or o_sf_scale is not None:
            raise ValueError("FA4 R0 target is full causal attention without sinks/output scaling")

        page_size = kv_cache.shape[-2]
        max_pages = (max_kv_len + page_size - 1) // page_size
        if max_pages > block_tables.shape[1]:
            raise ValueError("block_tables does not cover max_kv_len")

        # Zero-copy views from the production interleaved KV cache into FA4's
        # native paged NHD layout.  Supporting P64 in the dedicated HD256
        # kernel is deliberately left as optimization work, not hidden here.
        k_view = kv_cache[:, 0].transpose(1, 2)
        v_view = kv_cache[:, 1].transpose(1, 2)
        page_table = block_tables[:, :max_pages]
        returned = _flash_attn_fwd(
            query,
            k_view,
            v_view,
            cu_seqlens_q=cum_seq_lens_q,
            cu_seqlens_k=None,
            seqused_k=seq_lens,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_pages * page_size,
            page_table=page_table,
            softmax_scale=bmm1_scale,
            causal=True,
            num_splits=1,
            pack_gqa=True,
            out=out,
        )[0]
        if returned is not out:
            raise RuntimeError("FA4 private API did not preserve the supplied out tensor")
        return out
