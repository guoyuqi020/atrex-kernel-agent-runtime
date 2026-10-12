"""Reference for the observed non-FP8 GatedResidualSimple.combine call."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hc_count: int = 4, hidden_size: int = 2560) -> None:
        super().__init__()
        if type(hc_count) is not int or hc_count <= 0:
            raise ValueError("hc_count must be a positive integer")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.hc_count = hc_count
        self.hidden_size = hidden_size

    def forward(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        mix_aux: torch.Tensor,
    ) -> torch.Tensor:
        if (
            block_output.shape[:-1] != residual.shape[:-1]
            or block_output.shape[:-1] != mix_aux.shape[:-1]
        ):
            raise ValueError("combine batch dimensions differ")
        if (
            block_output.shape[-1] != self.hidden_size
            or residual.shape[-1] != self.hc_count * self.hidden_size
            or mix_aux.shape[-1] != self.hc_count
        ):
            raise ValueError("combine input width differs")
        expanded_residual = residual.unflatten(-1, (self.hc_count, self.hidden_size))
        injection = torch.einsum("...h,...c->...ch", block_output, mix_aux)
        return (expanded_residual + injection).flatten(-2)
