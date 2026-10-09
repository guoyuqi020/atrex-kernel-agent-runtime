"""Rank-local, value-independent BF16 communication input."""

from __future__ import annotations

import hashlib
import json
import os

import torch

ALLOWED_SHAPES = frozenset({
    (2, 2560),
    (4, 2560),
    (6, 2560),
    (8, 2560),
    (12, 2560),
    (14, 2560),
    (16, 2560),
    (64, 2560),
    (84, 2560),
    (110, 2560),
    (232, 2560),
    (234, 2560),
    (318, 2560),
    (362, 2560),
    (402, 2560),
    (432, 2560),
    (702, 2560),
    (708, 2560),
    (712, 2560),
    (714, 2560),
    (774, 2560),
    (1204, 2560),
    (1436, 2560),
    (7550, 2560),
    (8192, 2560),
})


def _make_inputs(
    shape: list[int], dtype: str = "bfloat16", device: str = "cuda"
) -> dict[str, torch.Tensor]:
    if tuple(shape) not in ALLOWED_SHAPES:
        raise ValueError("shape was not observed in TP2 capture")
    if dtype != "bfloat16":
        raise ValueError("TP2 capture used BF16")
    rank = int(os.environ["RANK"])
    if rank not in (0, 1) or int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("TP2 rank-local input requires two processes")
    shape_seed = int.from_bytes(
        hashlib.sha256(json.dumps(shape).encode()).digest()[:4], "little"
    )
    generator = torch.Generator(device="cpu").manual_seed(383800 + shape_seed + rank)
    x = torch.randn(tuple(shape), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    return {"x": x}
