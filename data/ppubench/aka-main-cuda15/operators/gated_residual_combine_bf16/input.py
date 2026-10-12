"""Generate combine inputs from three formal-compatible C16 real samples."""

from __future__ import annotations

import math

import torch


CALIBRATION_SHA256 = "a18b01ef07fa659e9c3797745a2cf9582e00671845f365e2626ea09f4abb0ce0"
_PROBABILITIES = (0.0, 0.01, 0.5, 0.99, 1.0)

# The two 4096-token captures have materially different scales, so they remain
# separate profiles. The 2-token capture provides the decode profile.
_PROFILES = {
    "prefill_narrow": {
        "block_output": (
            -0.35546875,
            -0.1318359375,
            0.0003509521484375,
            0.1435546875,
            1.125,
        ),
        "residual": (
            -0.1689453125,
            -0.023193359375,
            0.000133514404296875,
            0.0238037109375,
            0.14453125,
        ),
        "mix_aux": (
            1.5735626220703125e-05,
            6.628036499023438e-05,
            0.0299072265625,
            0.82421875,
            1.109375,
        ),
    },
    "prefill_wide": {
        "block_output": (-1.34375, -0.375, -0.00543212890625, 0.33984375, 0.890625),
        "residual": (
            -0.38671875,
            -0.05517578125,
            0.0001964569091796875,
            0.058264160156250355,
            0.271484375,
        ),
        "mix_aux": (
            2.0116567611694336e-06,
            1.1563301086425781e-05,
            0.0019989013671875,
            0.07047851562500007,
            1.5,
        ),
    },
    "decode": {
        "block_output": (-3.125, -1.09078125, 0.009765625, 1.0453906250000031, 4.625),
        "residual": (
            -6.65625,
            -1.618828125,
            0.004425048828125,
            1.6110156249999932,
            5.15625,
        ),
        "mix_aux": (0.55078125, 0.5516015625, 1.390625, 1.952578125, 1.953125),
    },
}
_INNER_EXPONENTS = {
    "prefill_narrow": {"block_output": 10.0, "residual": 24.0, "mix_aux": 2.8},
    "prefill_wide": {"block_output": 4.0, "residual": 24.0, "mix_aux": 3.5},
    "decode": {"block_output": 4.0, "residual": 5.5},
}

# Eight synthetic support points fitted to the 2-token mix_aux sample's
# quantiles, mean and population std; these are not copied tensor values.
_DECODE_MIX_SUPPORT = (0.5508, 0.5625, 0.7, 1.1, 1.68125, 1.75, 1.9453, 1.9531)


def _sample_quantiles(
    shape: tuple[int, ...],
    profile: tuple[float, float, float, float, float],
    *,
    seed: int,
    inner_exponent: float = 1.0,
) -> torch.Tensor:
    count = math.prod(shape)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if count <= 1024:
        uniform = (torch.arange(count, dtype=torch.float32) + 0.5) / count
        uniform = uniform[torch.randperm(count, generator=generator)]
    else:
        uniform = torch.rand(count, generator=generator)
    values = torch.empty_like(uniform)
    for index in range(4):
        lower = _PROBABILITIES[index]
        upper = _PROBABILITIES[index + 1]
        mask = (uniform >= lower) & (uniform < upper)
        if index == 3:
            mask |= uniform == upper
        fraction = (uniform[mask] - lower) / (upper - lower)
        if index == 0:
            fraction = fraction.pow(0.2)
        elif index == 1:
            fraction = 1.0 - (1.0 - fraction).pow(inner_exponent)
        elif index == 2:
            fraction = fraction.pow(inner_exponent)
        else:
            fraction = fraction.pow(8.0)
        values[mask] = profile[index] + fraction * (profile[index + 1] - profile[index])
    return values.reshape(shape)


def _sample_decode_mix_aux(shape: tuple[int, int], *, seed: int) -> torch.Tensor:
    count = math.prod(shape)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    support = torch.tensor(_DECODE_MIX_SUPPORT, dtype=torch.float32)
    positions = torch.linspace(0, len(support) - 1, count)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    values = support[lower] + (positions - lower) * (support[upper] - support[lower])
    return values[torch.randperm(count, generator=generator)].reshape(shape)


def _with_storage_offset(value: torch.Tensor, offset: int) -> torch.Tensor:
    if type(offset) is not int or offset < 0:
        raise ValueError("storage offset must be a non-negative integer")
    if offset == 0:
        return value
    storage = torch.zeros(
        offset + value.numel(), dtype=value.dtype, device=value.device
    )
    view = torch.as_strided(
        storage, size=value.shape, stride=value.stride(), storage_offset=offset
    )
    view.copy_(value)
    return view


def _make_inputs(
    tokens: int,
    hidden_size: int = 2560,
    hc_count: int = 4,
    profile: str = "prefill_narrow",
    dtype: str = "bfloat16",
    device: str = "cuda",
    residual_storage_offset: int = 0,
    mix_aux_storage_offset: int = 0,
) -> dict[str, torch.Tensor]:
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("tokens must be a positive integer")
    if hidden_size != 2560 or hc_count != 4 or dtype != "bfloat16":
        raise ValueError("combine geometry or dtype differs from the observed C16 ABI")
    if type(profile) is not str or profile not in _PROFILES:
        raise ValueError("combine numeric profile is not source-calibrated")
    quantiles = _PROFILES[profile]
    exponents = _INNER_EXPONENTS[profile]
    mix_aux = (
        _sample_decode_mix_aux((tokens, hc_count), seed=79)
        if profile == "decode"
        else _sample_quantiles(
            (tokens, hc_count),
            quantiles["mix_aux"],
            seed=79,
            inner_exponent=exponents["mix_aux"],
        )
    )
    return {
        "block_output": _sample_quantiles(
            (tokens, hidden_size),
            quantiles["block_output"],
            seed=71,
            inner_exponent=exponents["block_output"],
        ).to(device=device, dtype=torch.bfloat16),
        "residual": _with_storage_offset(
            _sample_quantiles(
                (tokens, hc_count * hidden_size),
                quantiles["residual"],
                seed=73,
                inner_exponent=exponents["residual"],
            ).to(device=device, dtype=torch.bfloat16),
            residual_storage_offset,
        ),
        "mix_aux": _with_storage_offset(
            mix_aux.to(device=device, dtype=torch.bfloat16), mix_aux_storage_offset
        ),
    }
