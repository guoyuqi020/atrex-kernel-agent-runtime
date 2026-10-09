"""Semantic reference for the Qwen3.8 fused GDN gating boundary."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    def forward(
        self, A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        softplus = F.softplus(a.float() + dt_bias.float(), beta=1.0, threshold=20.0)
        g = (-torch.exp(A_log.float()) * softplus).unsqueeze(0)
        beta_output = torch.sigmoid(b.float()).to(b.dtype).unsqueeze(0)
        return g, beta_output
