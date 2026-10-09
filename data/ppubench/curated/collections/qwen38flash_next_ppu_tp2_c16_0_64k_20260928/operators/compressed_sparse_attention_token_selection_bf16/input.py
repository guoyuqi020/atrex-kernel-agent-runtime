"""Construct source-valid packed CSA selector inputs from observed C16 shapes.

The Q/K value profiles are calibrated to three real pre-call tensors. Packed
row controls follow build_csa_row_ranges in the pinned Qwen3.8 vLLM source.
"""
from __future__ import annotations
import math
import torch
CALIBRATION_SHA256 = 'a25ba1e7a878d85c9755c6bd3cedde543e391c89327f680a8629ab3f68ed3f2e'
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
HEADS = 4
HEAD_DIM = 128
MAX_SEQUENCE_LENGTH = 65536

def _lengths(total: int, requests: int, minimum_active: int, active_index: int) -> list[int]:
    if total < requests or minimum_active > total - (requests - 1):
        raise ValueError('compressed key budget cannot support request controls')
    if requests == 1:
        return [total]
    weights = [1.0 + index % 5 * 0.17 for index in range(requests)]
    active = max(minimum_active, min(16383, round(total * 0.35)))
    active = min(active, total - requests + 1)
    remaining = total - active
    weights[active_index] = 0.0
    weight_sum = sum(weights)
    result = [1] * requests
    result[active_index] = active
    extras = remaining - (requests - 1)
    if extras < 0:
        raise ValueError('compressed key budget is too small')
    shares = [math.floor(extras * weight / weight_sum) for weight in weights]
    for index in range(requests):
        if index != active_index:
            result[index] += shares[index]
    unassigned = total - sum(result)
    for index in range(requests):
        if index != active_index and unassigned:
            result[index] += 1
            unassigned -= 1
    if unassigned or sum(result) != total:
        raise ValueError('compressed key partition did not conserve rows')
    return result

def _row_controls(query_rows: int, compressed_rows: int, profile: str, sequence_lengths: list[int] | None=None) -> dict[str, torch.Tensor]:
    supplied_lengths = sequence_lengths
    if profile == 'decode':
        if query_rows % 4 or not 4 <= query_rows <= 64:
            raise ValueError('decode selector rows must be 1–16 four-token requests')
        requests = query_rows // 4
        compressed_lengths = _lengths(compressed_rows, requests, 1, requests - 1)
        sequence_lengths = [4 * length + index % 4 for (index, length) in enumerate(compressed_lengths)]
        if supplied_lengths is not None:
            if not isinstance(supplied_lengths, list) or len(supplied_lengths) != requests or any((type(length) is not int for length in supplied_lengths)):
                raise TypeError('sequence lengths must contain one integer per request')
            if any((length < 4 or length // COMPRESS_RATIO != rows for (length, rows) in zip(supplied_lengths, compressed_lengths))):
                raise ValueError('sequence length differs from the compressed-row budget')
            sequence_lengths = supplied_lengths.copy()
        if any((length > MAX_SEQUENCE_LENGTH for length in sequence_lengths)):
            raise ValueError("decode context is outside the benchmark's 0–64K scope")
        token_to_req = torch.arange(requests, dtype=torch.int64, device='cpu').repeat_interleave(4)
        query_positions = torch.cat([torch.arange(sequence_lengths[index] - 4, sequence_lengths[index], device='cpu') for index in range(requests)]).to(torch.int64)
    elif profile == 'prefill':
        if supplied_lengths is not None:
            raise ValueError('explicit sequence lengths are currently supported for decode only')
        requests = 16 if compressed_rows >= 16 + query_rows // 4 else 1
        active = requests - 1
        compressed_lengths = _lengths(compressed_rows, requests, query_rows // 4, active)
        sequence_lengths = [4 * length + index % 4 for (index, length) in enumerate(compressed_lengths)]
        sequence_lengths[active] = 4 * compressed_lengths[active] + query_rows % 4
        if any((length > MAX_SEQUENCE_LENGTH for length in sequence_lengths)):
            raise ValueError("prefill context is outside the benchmark's 0–64K scope")
        token_to_req = torch.full((query_rows,), active, dtype=torch.int64, device='cpu')
        query_positions = torch.arange(sequence_lengths[active] - query_rows, sequence_lengths[active], dtype=torch.int64, device='cpu')
    else:
        raise ValueError('selector profile is not source-calibrated')
    compressed_lengths_tensor = torch.tensor(compressed_lengths, dtype=torch.int32, device='cpu')
    starts = torch.cat((torch.zeros(1, dtype=torch.int32, device='cpu'), compressed_lengths_tensor.cumsum(0)[:-1].to(torch.int32)))
    row_starts = starts.index_select(0, token_to_req)
    visible_blocks = torch.div(query_positions.to(torch.int32) + 1, COMPRESS_RATIO, rounding_mode='floor').clamp_min(0)
    max_blocks = compressed_lengths_tensor.index_select(0, token_to_req)
    row_ends = row_starts + torch.minimum(visible_blocks, max_blocks)
    sequence_lengths_rows = torch.tensor(sequence_lengths, dtype=torch.int32, device='cpu').index_select(0, token_to_req)
    if bool(torch.any(row_starts < 0) or torch.any(row_starts > row_ends) or torch.any(row_ends > compressed_rows)):
        raise ValueError('source-derived CSA row ranges are invalid')
    return {'row_starts': row_starts, 'row_ends': row_ends, 'query_positions': query_positions, 'sequence_lengths': sequence_lengths_rows}

def _make_inputs(query_rows: int, compressed_rows: int, profile: str, dtype: str='bfloat16', device: str='cuda', sequence_lengths: list[int] | None=None, seed: int | None=None) -> dict[str, object]:
    if type(query_rows) is not int or type(compressed_rows) is not int:
        raise TypeError('selector dimensions must be integers')
    if query_rows <= 0 or compressed_rows <= 0 or compressed_rows > 60000:
        raise ValueError('selector dimensions differ from the observed C16 range')
    if dtype != 'bfloat16':
        raise ValueError('selector dtype differs from the observed C16 ABI')
    controls = _row_controls(query_rows, compressed_rows, profile, sequence_lengths)
    generator = torch.Generator(device='cpu').manual_seed(torch.initial_seed() % 2 ** 63 if seed is None else seed)
    q = torch.randn((query_rows, HEADS, HEAD_DIM), generator=generator, dtype=torch.float32, device='cpu')
    keys = torch.randn((compressed_rows, 1, HEAD_DIM), generator=generator, dtype=torch.float32, device='cpu')
    q = (q * 1.045 - 0.02).to(dtype=torch.bfloat16, device=device)
    keys = (keys * 1.028 + 0.035).to(dtype=torch.bfloat16, device=device)
    return {'q': q, 'compressed_keys': keys, 'row_starts': controls['row_starts'].to(device=device), 'row_ends': controls['row_ends'].to(device=device), 'query_positions': controls['query_positions'].to(device=device), 'sequence_lengths': controls['sequence_lengths'].to(device=device), 'token_topk': TOKEN_TOPK, 'compress_ratio': COMPRESS_RATIO}
