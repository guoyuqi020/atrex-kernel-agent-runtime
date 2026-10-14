from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float = 0.0625,
        fa_version: int = 3,
    ) -> None:
        super().__init__()
        del fa_version
        self.max_seqlen_q = int(max_seqlen_q)
        self.max_seqlen_k = int(max_seqlen_k)
        self.softmax_scale = float(softmax_scale)

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        scheduler_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del q_descale, k_descale, v_descale, scheduler_metadata
        num_q_heads = q.shape[1]
        num_kv_heads = k_cache.shape[2]
        rep = num_q_heads // num_kv_heads
        block_size = k_cache.shape[1]
        out = torch.zeros_like(q)
        q_bounds = cu_seqlens_q.tolist()
        kv_lens = seqused_k.tolist()
        for request_id, kv_len in enumerate(kv_lens):
            qs, qe = int(q_bounds[request_id]), int(q_bounds[request_id + 1])
            q_len = qe - qs
            if q_len <= 0:
                continue
            num_pages = (int(kv_len) + block_size - 1) // block_size
            page_ids = block_table[request_id, :num_pages].long()
            k_req = k_cache[page_ids].reshape(-1, num_kv_heads, q.shape[-1])[:kv_len]
            v_req = v_cache[page_ids].reshape(-1, num_kv_heads, q.shape[-1])[:kv_len]
            qf = q[qs:qe].transpose(0, 1).float()
            kf = k_req.repeat_interleave(rep, dim=1).transpose(0, 1).float()
            vf = v_req.repeat_interleave(rep, dim=1).transpose(0, 1).float()
            scores = torch.matmul(qf, kf.transpose(-1, -2)) * self.softmax_scale
            i = torch.arange(q_len, device=q.device).unsqueeze(-1)
            j = torch.arange(int(kv_len), device=q.device).unsqueeze(0)
            mask = j <= (int(kv_len) - q_len + i)
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out[qs:qe] = torch.matmul(attn, vf).transpose(0, 1).to(q.dtype)
        return out
