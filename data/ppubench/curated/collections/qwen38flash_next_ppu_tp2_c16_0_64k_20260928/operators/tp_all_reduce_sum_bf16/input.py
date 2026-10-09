"""Rank-local, value-independent BF16 communication input."""

from __future__ import annotations

import hashlib
import json
import os

import torch

ALLOWED_SHAPES = frozenset({
    (1, 2560),
    (2, 2560),
    (3, 2560),
    (4, 16, 160),
    (4, 2560),
    (7, 2560),
    (8, 16, 160),
    (8, 2560),
    (14, 2560),
    (15, 2560),
    (28, 16, 160),
    (28, 2560),
    (231, 16, 160),
    (318, 2560),
    (402, 2560),
    (701, 16, 160),
    (707, 2560),
    (714, 16, 160),
    (1204, 16, 160),
    (1757, 16, 160),
    (1757, 2560),
    (2269, 16, 160),
    (2269, 2560),
    (8192, 16, 160),
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
