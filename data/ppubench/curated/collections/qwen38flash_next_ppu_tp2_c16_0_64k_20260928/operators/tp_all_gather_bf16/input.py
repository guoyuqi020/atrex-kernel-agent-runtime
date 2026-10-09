"""Rank-local, value-independent BF16 communication input."""

from __future__ import annotations

import hashlib
import json
import os

import torch

ALLOWED_SHAPES = frozenset({
    (4096, 2560),
    (32, 2560),
    (2, 2560),
    (4, 2560),
    (42, 2560),
    (55, 2560),
    (8, 2560),
    (1, 2560),
    (6, 2560),
    (4096, 10240),
    (7, 2560),
    (32, 10240),
    (8, 10240),
    (1, 10240),
    (2, 10240),
    (4, 10240),
    (159, 2560),
    (159, 10240),
    (116, 2560),
    (117, 2560),
    (16, 4, 1280),
    (16, 1280),
    (1, 4, 1280),
    (1, 1280),
    (234, 1280),
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
