from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

_BLOCK_M = 8
_CURRENT_PAD_SLOT_ID = -1


@dataclass(frozen=True)
class FlattenedPrefillSchedule:
    """Portable projection of source metadata scheduling fields.

    It carries the source ``batch_ptr`` and ``token_chunk_offset_ptr`` launch
    arrays after flattening ``GDNAttentionMetadata.nums_dict`` for ``BLOCK_M=8``.
    The opaque runtime metadata object is deliberately not accepted directly.
    """

    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor


def _normalized_activation(activation: bool | str | None) -> str | None:
    if isinstance(activation, bool):
        return "silu" if activation else None
    if activation in (None, "silu", "swish"):
        return activation
    raise ValueError("activation must be None, 'silu', or 'swish'")


def _sequence_bounds(query_start_loc: torch.Tensor, total_tokens: int) -> list[tuple[int, int]]:
    if query_start_loc.dtype != torch.int32 or query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must be one-dimensional int32")
    starts = [int(value) for value in query_start_loc.tolist()]
    if len(starts) < 2 or starts[0] != 0 or starts[-1] != total_tokens:
        raise ValueError("query_start_loc must start at zero and end at x token count")
    if any(start > end for start, end in pairwise(starts)):
        raise ValueError("query_start_loc must be monotonic")
    return list(pairwise(starts))


def _validate_schedule(
    metadata: FlattenedPrefillSchedule,
    bounds: list[tuple[int, int]],
    pad_slot_id: int,
) -> None:
    batch_ptr = metadata.batch_ptr
    token_chunk_offset_ptr = metadata.token_chunk_offset_ptr
    if (
        batch_ptr.dtype != torch.int32
        or token_chunk_offset_ptr.dtype != torch.int32
        or batch_ptr.ndim != 1
        or token_chunk_offset_ptr.ndim != 1
        or batch_ptr.numel() != token_chunk_offset_ptr.numel()
    ):
        raise ValueError("flattened schedule requires equally sized one-dimensional int32 tensors")

    expected = {
        (sequence, chunk)
        for sequence, (start, end) in enumerate(bounds)
        for chunk in range((end - start + _BLOCK_M - 1) // _BLOCK_M)
    }
    observed: set[tuple[int, int]] = set()
    for sequence, chunk_offset in zip(batch_ptr.tolist(), token_chunk_offset_ptr.tolist()):
        if sequence == pad_slot_id:
            continue
        pair = (int(sequence), int(chunk_offset))
        if pair not in expected:
            raise ValueError("flattened schedule contains an out-of-contract sequence/chunk pair")
        if pair in observed:
            raise ValueError("flattened schedule repeats a sequence/chunk pair")
        observed.add(pair)
    if observed != expected:
        raise ValueError("flattened schedule does not cover every source sequence chunk")


def _convolve_sequence(
    initial_state: torch.Tensor,
    sequence: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    width=weight.shape[1]
    packed=torch.cat((initial_state.float(),sequence.float()),dim=1)
    output=torch.empty_like(sequence,dtype=torch.float32)
    for start in range(0,sequence.shape[1],2048):
        stop=min(start+2048,sequence.shape[1])
        windows=packed[:,start:stop+width-1].unfold(1,width,1)
        products=windows*weight.float()[:,None,:]
        # The pinned Triton source multiplies two BF16 loads before extending
        # that product into its FP32 accumulator; preserve this rounding point.
        if sequence.dtype==torch.bfloat16:products=products.to(torch.bfloat16).float()
        values=torch.zeros_like(products[...,0],dtype=torch.float32)
        if bias is not None:values=values+bias.float()[:,None]
        for tap in range(width):values.add_(products[...,tap])
        if activation is not None:values=F.silu(values)
        output[:,start:stop]=values
    return output,packed[:,-(width-1):]



class Model(nn.Module):
    """Portable vectorized math reference of packed ``causal_conv1d_fn`` without APC.

    Source binding: ``causal_conv1d.py`` SHA256
    ``815b4db17be081fca5159ed444abea259195ae9d5f027c8ec074745f5c9259ed``;
    the prefill contract is lines 495-760, especially state length and masks
    at 72-74, 161-177, 238, 591, and 614.  Only leading ``width - 1`` cache
    positions participate in this prefill draft. Physical cache tail capacity
    remains unchanged.
    """

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
        activation: str | None = "silu",
        pad_slot_id: int = _CURRENT_PAD_SLOT_ID,
        block_idx_first_scheduled_token: torch.Tensor | None = None,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        block_size_to_align: int = 0,
        metadata: FlattenedPrefillSchedule | None = None,
        validate_data: bool = False,
    ) -> torch.Tensor:
        if any(
            value is not None
            for value in (
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
                initial_state_idx,
                num_computed_tokens,
            )
        ):
            raise NotImplementedError("APC scheduling is not enabled in the captured contract")
        if block_size_to_align != 0:
            raise NotImplementedError("nonzero block_size_to_align is outside captured contract")
        if validate_data:
            raise NotImplementedError("validate_data=True is outside captured prefill contract")
        if cache_indices is None or has_initial_state is None:
            raise ValueError("prefill requires cache_indices and has_initial_state")
        if x.ndim != 2 or weight.ndim != 2 or conv_states.ndim != 3:
            raise ValueError("x, weight, and conv_states must be rank 2, 2, and 3")
        if weight.size(0) != x.size(0) or conv_states.size(1) != x.size(0):
            raise ValueError("feature dimensions of x, weight, and conv_states must agree")
        if weight.stride(1) != 1 or conv_states.stride(1) != 1:
            raise ValueError("layout requires contiguous weight width and cache feature axes")
        if bias is not None and tuple(bias.shape) != (x.size(0),):
            raise ValueError("bias must be None or have shape [dim]")
        if cache_indices.dtype != torch.int32 or cache_indices.ndim != 1:
            raise ValueError("cache_indices must be one-dimensional int32")
        if has_initial_state.dtype != torch.bool or has_initial_state.ndim != 1:
            raise ValueError("has_initial_state must be one-dimensional bool")

        bounds = _sequence_bounds(query_start_loc, x.size(1))
        if cache_indices.numel() != len(bounds) or has_initial_state.numel() != len(bounds):
            raise ValueError("cache_indices and has_initial_state must have one value per sequence")
        state_len = weight.size(1) - 1
        if state_len < 1 or conv_states.size(2) < state_len:
            raise ValueError("conv_states must hold leading width - 1 state positions")
        if metadata is not None:
            if not isinstance(metadata, FlattenedPrefillSchedule):
                raise TypeError(
                    "opaque metadata requires a source-bound FlattenedPrefillSchedule adapter"
                )
            _validate_schedule(metadata, bounds, pad_slot_id)

        normalized_activation = _normalized_activation(activation)
        original_x_dtype = x.dtype
        working_x = x.to(conv_states.dtype)
        output = torch.empty_like(working_x)
        for sequence_index, (start, end) in enumerate(bounds):
            if start == end:
                continue
            cache_index = int(cache_indices[sequence_index].item())
            if cache_index == pad_slot_id:
                continue
            if cache_index < 0 or cache_index >= conv_states.size(0):
                raise ValueError("cache index is outside conv_states")
            if bool(has_initial_state[sequence_index].item()):
                initial_state = conv_states[cache_index, :, :state_len]
            else:
                initial_state = torch.zeros(
                    (x.size(0), state_len), dtype=conv_states.dtype, device=x.device
                )
            values, final_state = _convolve_sequence(
                initial_state,
                working_x[:, start:end],
                weight,
                bias,
                normalized_activation,
            )
            output[:, start:end].copy_(values.to(output.dtype))
            conv_states[cache_index, :, :state_len].copy_(final_state.to(conv_states.dtype))
        return output.to(original_x_dtype)
