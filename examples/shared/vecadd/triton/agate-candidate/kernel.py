"""Shared Triton vector-add candidate for direct Agate operations."""

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add(
    left,
    right,
    output,
    length,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < length
    values = tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


class Model(torch.nn.Module):
    """Atrex-Bench candidate entrypoint."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(left)
        length = left.numel()
        grid = (triton.cdiv(length, 256),)
        _vector_add[grid](left, right, output, length, BLOCK_SIZE=256)
        return output
