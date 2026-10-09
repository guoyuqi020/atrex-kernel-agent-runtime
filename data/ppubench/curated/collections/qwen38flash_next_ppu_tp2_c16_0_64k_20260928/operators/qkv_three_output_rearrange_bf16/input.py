"""Construct C16 QKV rearrangement inputs calibrated to three real tensors."""

from __future__ import annotations

import torch


CALIBRATION_SHA256 = "a18b01ef07fa659e9c3797745a2cf9582e00671845f365e2626ea09f4abb0ce0"
WIDTH = 5120


def _make_inputs(
    tokens: int,
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    if type(tokens) is not int or not 1 <= tokens <= 8192:
        raise ValueError("QKV token count is outside the observed C16 active range")
    if dtype != "bfloat16":
        raise ValueError("QKV dtype differs from the observed C16 ABI")
    generator = torch.Generator(device="cpu").manual_seed(12097 + tokens)
    # Captured BF16 inputs have a narrow core (p01 around -0.21, p99 around
    # 0.28) and rare positive outliers up to 7.31; Gaussian(0, 1) is not
    # representative of either tail.
    core = torch.randn((tokens, WIDTH), generator=generator, dtype=torch.float32)
    core = (core * 0.105 + 0.016).clamp_(-0.28, 0.32)
    spikes = torch.rand((tokens, WIDTH), generator=generator) < 0.001
    values = torch.where(spikes, 4.5 + 2.8 * torch.rand(
        (tokens, WIDTH), generator=generator
    ), core)
    return {"mixed_qkv": values.to(dtype=torch.bfloat16, device=device)}
