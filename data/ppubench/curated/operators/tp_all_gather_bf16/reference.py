"""Mathematical TP2 reference using broadcast, independent of production collective."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class Model(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim not in (0, -1):
            raise ValueError("unsupported AllGather dimension")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized() or dist.get_world_size() != 2:
            raise RuntimeError("TP2 reference requires a paired process group")
        rank = dist.get_rank()
        parts = []
        for source_rank in range(2):
            value = x.clone() if rank == source_rank else torch.empty_like(x)
            dist.broadcast(value, src=source_rank)
            parts.append(value)
        return torch.cat(parts, dim=self.dim)
