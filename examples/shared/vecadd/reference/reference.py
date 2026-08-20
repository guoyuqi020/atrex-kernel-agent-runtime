"""Shared trusted reference for the vector-add examples."""

import torch


class Model(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right
