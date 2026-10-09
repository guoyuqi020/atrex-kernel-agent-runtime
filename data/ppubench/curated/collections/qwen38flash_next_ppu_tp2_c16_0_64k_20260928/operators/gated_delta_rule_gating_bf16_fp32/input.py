"""Construct C16 GDN gating inputs from three real numerical samples."""

from __future__ import annotations

import torch


CALIBRATION_SHA256 = "a18b01ef07fa659e9c3797745a2cf9582e00671845f365e2626ea09f4abb0ce0"
HEADS = 24


def _make_inputs(
    tokens: int,
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    if type(tokens) is not int or not 1 <= tokens <= 8192:
        raise ValueError("gating token count is outside the observed C16 range")
    if dtype != "bfloat16":
        raise ValueError("gating activation dtype differs from the observed ABI")
    # A_log and dt_bias are fixed model parameters, so they stay identical
    # across token-count cases. Means and standard deviations are from real
    # pre-call tensors, not the old synthetic 0.1-scale benchmark.
    parameter_rng = torch.Generator(device="cpu").manual_seed(383801)
    activation_rng = torch.Generator(device="cpu").manual_seed(383802 + tokens)
    a_log = torch.randn((HEADS,), generator=parameter_rng) * 2.75 + 1.275
    dt_bias = (torch.randn((HEADS,), generator=parameter_rng) * 2.467 - 2.953)
    a = torch.randn((tokens, HEADS), generator=activation_rng) * 2.95 - 0.4
    b = torch.randn((tokens, HEADS), generator=activation_rng) * 2.3 + 1.77
    return {
        "A_log": a_log.to(dtype=torch.float32, device=device),
        "a": a.to(dtype=torch.bfloat16, device=device),
        "b": b.to(dtype=torch.bfloat16, device=device),
        "dt_bias": dt_bias.to(dtype=torch.bfloat16, device=device),
    }
