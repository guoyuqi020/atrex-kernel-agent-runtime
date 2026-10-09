"""Portable source-bound mathematical reference for Qwen4-Exp speculative GDN decode.

This draft models only the observed packed speculative branch that calls
``fused_recurrent_gated_delta_rule`` with an in-place FP32 state pool.  It is
a PyTorch implementation and deliberately rejects source-supported paths that
were absent from the captured call (for example EAGLE parent-token retrieval).
"""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import nn

_BATCH_SIZE = 1
_KEY_HEADS = 8
_VALUE_HEADS = 24
_KEY_DIM = 128
_VALUE_DIM = 128
_HEAD_GROUP = _VALUE_HEADS // _KEY_HEADS
_L2NORM_EPSILON = 1e-6
_STATE_POOL_STRIDE = 28 * _VALUE_DIM * _KEY_DIM
_VALUE_TO_KEY_HEAD = torch.arange(_VALUE_HEADS, dtype=torch.long, device="cpu") // _HEAD_GROUP

# The independently accepted C16 ABI catalog observed these packed speculative decode pairs:
# (number of sequences, total flattened tokens), with four token positions per
# sequence and a [N, 4] state-index table.
OBSERVED_SPECULATIVE_GEOMETRIES = tuple((count, 4 * count) for count in range(1, 17))


def source_kernel_l2norm(x: torch.Tensor) -> torch.Tensor:
    """Mirror the fused kernel's FP32 q/k normalization before score scaling."""

    if x.dtype != torch.bfloat16:
        raise ValueError("q/k normalization input must be bfloat16")
    if x.ndim < 1 or x.shape[-1] != _KEY_DIM:
        raise ValueError("q/k normalization input must end in K=128")
    values = x.float()
    return values * torch.rsqrt(values.square().sum(dim=-1, keepdim=True) + _L2NORM_EPSILON)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_contiguous(name: str, value: torch.Tensor) -> None:
    _require(value.is_contiguous(), f"{name} must be contiguous in the observed decode ABI")


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
) -> tuple[int, list[int], list[int]]:
    _require(q.dtype == torch.bfloat16, "q must be bfloat16")
    _require(k.dtype == torch.bfloat16, "k must be bfloat16")
    _require(v.dtype == torch.bfloat16, "v must be bfloat16")
    _require(beta.dtype == torch.bfloat16, "beta must be bfloat16")
    _require(g.dtype == torch.float32, "g must be float32")
    _require(initial_state.dtype == torch.float32, "initial_state must be float32")
    for name, value in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        _require(value.device == q.device, f"{name} must share q device")
        _require_contiguous(name, value)

    _require(q.ndim == 4, "q must have shape [1, T, 8, 128]")
    _require(tuple(k.shape) == tuple(q.shape), "k must match q shape")
    _require(q.shape[0] == _BATCH_SIZE, "packed speculative decode requires batch size 1")
    _require(q.shape[2:] == (_KEY_HEADS, _KEY_DIM), "q/k must use Hq=8 and K=128")
    tokens = q.shape[1]
    _require(
        tuple(v.shape) == (_BATCH_SIZE, tokens, _VALUE_HEADS, _VALUE_DIM),
        "v must have shape [1, T, 24, 128]",
    )
    _require(
        tuple(g.shape) == (_BATCH_SIZE, tokens, _VALUE_HEADS),
        "g must have shape [1, T, 24]",
    )
    _require(
        tuple(beta.shape) == (_BATCH_SIZE, tokens, _VALUE_HEADS),
        "beta must have shape [1, T, 24]",
    )

    _require(cu_seqlens is not None, "cu_seqlens is required for packed speculative decode")
    _require(cu_seqlens.dtype == torch.int32, "cu_seqlens must be int32")
    _require(cu_seqlens.ndim == 1 and cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous and one-dimensional")
    _require(cu_seqlens.numel() >= 2, "cu_seqlens must contain at least one sequence")
    bounds = [int(value) for value in cu_seqlens.tolist()]
    sequence_count = len(bounds) - 1
    _require(bounds[0] == 0 and bounds[-1] == tokens, "cu_seqlens must span all flattened tokens")
    _require(
        all(end - start == 4 for start, end in pairwise(bounds)),
        "each observed speculative sequence must contain four token positions",
    )
    _require(
        (sequence_count, tokens) in OBSERVED_SPECULATIVE_GEOMETRIES,
        "inputs must use a canonical speculative geometry from the sealed catalog",
    )

    _require(ssm_state_indices is not None, "ssm_state_indices is required for speculative decode")
    _require(ssm_state_indices.dtype == torch.int32, "ssm_state_indices must be int32")
    _require(
        ssm_state_indices.ndim == 2 and ssm_state_indices.is_contiguous(),
        "ssm_state_indices must be contiguous with shape [N, 4]",
    )
    _require(
        tuple(ssm_state_indices.shape) == (sequence_count, 4),
        "ssm_state_indices must have shape [N, 4]",
    )

    _require(num_accepted_tokens is not None, "num_accepted_tokens is required for speculative decode")
    _require(num_accepted_tokens.dtype == torch.int32, "num_accepted_tokens must be int32")
    _require(
        num_accepted_tokens.ndim == 1 and num_accepted_tokens.is_contiguous(),
        "num_accepted_tokens must be contiguous and one-dimensional",
    )
    _require(
        num_accepted_tokens.numel() == sequence_count,
        "num_accepted_tokens must contain one value per sequence",
    )
    accepted = [int(value) for value in num_accepted_tokens.tolist()]
    _require(
        all(1 <= count <= 4 for count in accepted),
        "num_accepted_tokens must be in [1, 4] for each speculative sequence",
    )

    _require(all(value.device == q.device for value in (initial_state, cu_seqlens, ssm_state_indices, num_accepted_tokens)), "state and controls must share q device")
    _require(initial_state.ndim == 4, "initial_state must have shape [P, 24, 128, 128]")
    _require(initial_state.shape[0] > 0, "initial_state pool must contain at least one slot")
    _require(
        tuple(initial_state.shape[1:]) == (_VALUE_HEADS, _VALUE_DIM, _KEY_DIM),
        "initial_state must have shape [P, 24, 128, 128]",
    )
    _require(
        initial_state.stride(-1) == 1
        and initial_state.stride(-2) == _KEY_DIM
        and initial_state.stride(-3) == _VALUE_DIM * _KEY_DIM
        and initial_state.stride(0) == _STATE_POOL_STRIDE,
        "initial_state must preserve the observed 28-head padded outer stride and [V, K] inner layout",
    )

    _require(initial_state.storage_offset() == 15360, "initial_state must retain the observed storage offset")

    slots = [int(value) for value in ssm_state_indices.flatten().tolist()]
    _require(all(slot >= -1 for slot in slots), "ssm_state_indices may use only -1 as PAD_SLOT_ID")
    valid_slots = [slot for slot in slots if slot >= 0]
    _require(
        all(slot < initial_state.shape[0] for slot in valid_slots),
        "ssm_state_indices contains a state slot outside the state pool",
    )
    _require(
        len(valid_slots) == len(set(valid_slots)),
        "valid speculative state slots must be unique for deterministic CPU semantics",
    )
    for sequence, count in enumerate(accepted):
        selected_slot = int(ssm_state_indices[sequence, count - 1])
        _require(
            selected_slot >= 0,
            "selected initial state may not be PAD_SLOT_ID in this CPU draft",
        )
    return sequence_count, bounds, accepted


class Model(nn.Module):
    """Batched math golden for the captured in-place speculative decode branch only.

    ``final_state`` aliases and mutates ``initial_state`` exactly as the source
    path does.  The source's EAGLE retrieval path and non-speculative decode
    call omit the observed `num_accepted_tokens` contract, so they are excluded.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        inplace_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not inplace_final_state:
            raise ValueError("inplace_final_state must be true for the captured decode ABI")
        if retrieve_parent_token is not None:
            raise NotImplementedError(
                "retrieve_parent_token is not present in the captured speculative decode ABI"
            )
        if not use_qk_l2norm_in_kernel:
            raise ValueError(
                "use_qk_l2norm_in_kernel must be true for the captured decode ABI"
            )

        _sequence_count, bounds, accepted = _validate_inputs(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
        )
        assert ssm_state_indices is not None

        head_map = _VALUE_TO_KEY_HEAD.to(q.device)
        normalized_q = source_kernel_l2norm(q).squeeze(0).index_select(1, head_map)
        normalized_k = source_kernel_l2norm(k).squeeze(0).index_select(1, head_map)
        q_values = normalized_q * math.pow(_KEY_DIM, -0.5)
        v_values = v.squeeze(0).float()
        g_values = g.squeeze(0)
        beta_values = beta.squeeze(0).float()
        output = torch.empty(
            (_BATCH_SIZE, q.shape[1], _VALUE_HEADS, _VALUE_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )

        # Read every initial state before any pool writes. Valid destination
        # slots are unique, so four batched steps preserve the serial semantics.
        indices = ssm_state_indices.long()
        accepted_index = torch.tensor(accepted,device=q.device,dtype=torch.long)-1
        sequence_index = torch.arange(_sequence_count,device=q.device)
        state = initial_state[indices[sequence_index,accepted_index]].clone()
        keys = normalized_k.reshape(_sequence_count,4,_VALUE_HEADS,_KEY_DIM)
        queries = q_values.reshape(_sequence_count,4,_VALUE_HEADS,_KEY_DIM)
        values = v_values.reshape(_sequence_count,4,_VALUE_HEADS,_VALUE_DIM)
        gates = g_values.reshape(_sequence_count,4,_VALUE_HEADS)
        weights = beta_values.reshape(_sequence_count,4,_VALUE_HEADS)
        packed_output = output[0].view(_sequence_count,4,_VALUE_HEADS,_VALUE_DIM)
        for position in range(4):
            state *= gates[:,position].exp()[:,:,None,None]
            key = keys[:,position]
            reconstructed = (state*key[:,:,None,:]).sum(-1)
            delta = (values[:,position]-reconstructed)*weights[:,position,:,None]
            state += delta[:,:,:,None]*key[:,:,None,:]
            packed_output[:,position] = (state*queries[:,position,:,None,:]).sum(-1).to(torch.bfloat16)
            destinations = indices[:,position]
            valid = destinations >= 0
            initial_state[destinations[valid]] = state[valid]
        return output, initial_state
