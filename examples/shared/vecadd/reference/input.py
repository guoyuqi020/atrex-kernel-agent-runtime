"""Shared Atrex-Bench input generator for the vector-add examples."""

import torch


def _make_inputs(num_elements: int) -> dict[str, torch.Tensor]:
    """Build keyword arguments matching ``Model.forward(left, right)``."""
    left = torch.randn((num_elements,), device="cuda", dtype=torch.float32)
    return {"left": left, "right": torch.randn_like(left)}
