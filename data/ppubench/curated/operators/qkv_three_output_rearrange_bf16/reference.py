"""Exact TP2 Qwen3.8 GDN rearrange_mixed_qkv tensor boundary."""

from __future__ import annotations

import torch
from torch import nn


class Model(nn.Module):
    def forward(self, mixed_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = torch.split(mixed_qkv, [1024, 1024, 3072], dim=-1)
        query, key = map(
            lambda x: x.unflatten(-1, (-1, 128)).unsqueeze(0),
            (query, key),
        )
        value = value.unflatten(-1, (-1, 128)).unsqueeze(0)
        return query.contiguous(), key.contiguous(), value.contiguous()
