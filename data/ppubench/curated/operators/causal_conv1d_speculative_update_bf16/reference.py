from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

_CURRENT_PAD_SLOT_ID = -1


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


def _convolve_sequence(
    initial_state: torch.Tensor,
    sequence: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
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
    return output



class Model(nn.Module):
    """Portable vectorized math reference of captured varlen speculative ``causal_conv1d_update``.

    Source binding: ``causal_conv1d.py`` SHA256
    ``815b4db17be081fca5159ed444abea259195ae9d5f027c8ec074745f5c9259ed``,
    lines 1141-1326 and kernel state logic 855-952.  It models current
    ``num_accepted_tokens`` plus ``query_start_loc`` use only. APC and EAGLE
    tree branches are explicitly outside the captured ABI.
    """

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: bool | str | None = None,
        conv_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        max_query_len: int = -1,
        pad_slot_id: int = _CURRENT_PAD_SLOT_ID,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        validate_data: bool = False,
    ) -> torch.Tensor:
        if block_idx_last_scheduled_token is not None or initial_state_idx is not None:
            raise NotImplementedError("APC scheduling is not enabled in the captured contract")
        if retrieve_parent_token is not None:
            raise NotImplementedError("EAGLE retrieve_parent_token is outside captured contract")
        if validate_data:
            raise NotImplementedError("validate_data=True is outside captured decode contract")
        if query_start_loc is None or conv_state_indices is None or num_accepted_tokens is None:
            raise ValueError(
                "decode requires query_start_loc, state indices, and accepted-token counts"
            )
        if x.ndim != 2 or conv_state.ndim != 3 or weight.ndim != 2:
            raise ValueError("decode requires rank-2 x, rank-3 state, and rank-2 weight")
        if weight.size(0) != x.size(1) or conv_state.size(1) != x.size(1):
            raise ValueError("feature dimensions of x, weight, and conv_state must agree")
        if weight.stride(1) != 1 or conv_state.stride(1) != 1:
            raise ValueError("layout requires contiguous weight width and cache feature axes")
        if bias is not None and tuple(bias.shape) != (x.size(1),):
            raise ValueError("bias must be None or have shape [dim]")
        if (
            conv_state_indices.dtype != torch.int32
            or num_accepted_tokens.dtype != torch.int32
            or conv_state_indices.ndim != 1
            or num_accepted_tokens.ndim != 1
        ):
            raise ValueError("state indices and accepted counts must be 1D int32")

        bounds = _sequence_bounds(query_start_loc, x.size(0))
        batch = len(bounds)
        if conv_state_indices.numel() != batch or num_accepted_tokens.numel() != batch:
            raise ValueError("state indices and accepted counts need one value per sequence")
        if max_query_len < 1 or max_query_len < max(end - start for start, end in bounds):
            raise ValueError("max_query_len must cover every actual query length")
        width = weight.size(1)
        if width < 2:
            raise ValueError("captured causal convolution requires width at least two")
        required_capacity = width - 1 + (max_query_len - 1)
        if conv_state.size(2) < required_capacity:
            raise ValueError("conv_state lacks capacity for the captured speculative state window")

        normalized_activation = _normalized_activation(activation)
        original_x_dtype = x.dtype
        working_x = x.to(conv_state.dtype)
        output = torch.empty_like(working_x)
        for sequence_index, (start, end) in enumerate(bounds):
            actual_length = end - start
            if actual_length == 0:
                continue
            cache_index = int(conv_state_indices[sequence_index].item())
            if cache_index == pad_slot_id:
                continue
            if cache_index < 0 or cache_index >= conv_state.size(0):
                raise ValueError("cache index is outside conv_state")
            accepted_tokens = int(num_accepted_tokens[sequence_index].item())
            if not 1 <= accepted_tokens <= actual_length:
                raise ValueError("accepted-token count must be within the actual query length")
            effective_state_len = width - 1 + (actual_length - 1)
            state_offset = accepted_tokens - 1
            initial_state = conv_state[cache_index, :, state_offset : state_offset + width - 1]
            sequence = working_x[start:end].transpose(0, 1)
            values = _convolve_sequence(
                initial_state,
                sequence,
                weight,
                bias,
                normalized_activation,
            )
            output[start:end].copy_(values.transpose(0, 1).to(output.dtype))

            retained_history = conv_state[
                cache_index, :, accepted_tokens : accepted_tokens + width - 2
            ]
            updated_state = torch.cat((retained_history, sequence.float()), dim=1)
            conv_state[cache_index, :, :effective_state_len].copy_(
                updated_state.to(conv_state.dtype)
            )
        return output.to(original_x_dtype)
