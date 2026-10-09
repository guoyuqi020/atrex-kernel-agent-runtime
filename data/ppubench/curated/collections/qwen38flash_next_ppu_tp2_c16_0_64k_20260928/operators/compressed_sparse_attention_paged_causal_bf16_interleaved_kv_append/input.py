"""Source-valid CSA backend state for observed C16 PPU geometries.

Q/K/V profiles come from three real pre-call captures. The large cache keeps
the production interleaved stride; only logical prior-token slots are filled.
"""
from __future__ import annotations
import math
import torch
CALIBRATION_SHA256 = '095cd1266506948440003dac1778de4bff46267ecc32d4a7ba0ed8ac682bf2c2'
ACTIVE_CACHE_SHA256 = '5fa9620f177b32e4b2e693f726e097224b670ccaafc009f3f6a35e137754fb36'
CACHE_BLOCKS = 1282
BLOCK_SIZE = 1792
HEAD_DIM = 256
QUERY_HEADS = 12
INDEX_WIDTH = 2051
MAX_QUERY_TOKENS = 8192
TABLE_WIDTH = 586
_DRAFT_QUANTILES = {'query': (-83.0, -10.25, -0.065673828125, 7.5625, 21.125), 'key': (-87.0, -4.409375190734863, -0.05419921875, 3.015625, 4.875), 'value': (-4.75, -2.90625, 0.0751953125, 2.921875, 4.59375)}

def _profile_tensor(shape: tuple[int, ...], name: str, profile: str, generator: torch.Generator) -> torch.Tensor:
    if profile == 'draft_decode':
        quantiles = _DRAFT_QUANTILES[name]
        tail_exponent = {'query': 0.5, 'key': 4.0, 'value': 0.05}[name]
        inner_exponent = {'query': 8.0, 'key': 8.0, 'value': 2.0}[name]
        uniforms = torch.rand(shape, generator=generator, dtype=torch.float32, device='cpu')
        values = torch.empty_like(uniforms)
        cutoffs = (0.0, 0.01, 0.5, 0.99, 1.0)
        for index in range(4):
            (lower, upper) = (cutoffs[index], cutoffs[index + 1])
            mask = (uniforms >= lower) & (uniforms < upper)
            fraction = (uniforms[mask] - lower) / (upper - lower)
            if index == 0:
                fraction = fraction.pow(tail_exponent)
            elif index == 1:
                fraction = 1.0 - (1.0 - fraction).pow(inner_exponent)
            elif index == 2:
                fraction = fraction.pow(inner_exponent)
            elif index == 3:
                fraction = fraction.pow(4.0)
            values[mask] = quantiles[index] + fraction * (quantiles[index + 1] - quantiles[index])
        return values
    if name == 'query':
        (mean, std) = (0.049, 1.178) if profile == 'main_prefill' else (0.014, 1.214)
    elif name == 'key':
        (mean, std) = (-0.018, 1.348) if profile == 'main_prefill' else (-0.057, 1.35)
    else:
        (mean, std) = (0.016, 0.48) if profile == 'main_prefill' else (0.028, 0.489)
    return torch.randn(shape, generator=generator, dtype=torch.float32, device='cpu') * std + mean

def _query_lengths(query_tokens: int, requests: int, profile: str) -> list[int]:
    if profile == 'main_prefill':
        if requests == 1:
            return [query_tokens]
        if query_tokens != MAX_QUERY_TOKENS or not 11 <= requests <= 16:
            raise ValueError('prefill geometry differs from selected C16 Shapes')
        return [0] * (requests - 1) + [query_tokens]
    if profile == 'main_decode':
        if query_tokens != 4 * requests:
            raise ValueError('main decode must have four query tokens per request')
        return [4] * requests
    if profile == 'draft_decode':
        if not requests <= query_tokens <= 4 * requests:
            raise ValueError('draft decode query/request geometry differs')
        result = [1] * requests
        remaining = query_tokens - requests
        for index in range(requests):
            take = min(3, remaining)
            result[index] += take
            remaining -= take
        if remaining:
            raise ValueError('draft decode query tokens are not conserved')
        return result
    raise ValueError('CSA profile is not source-calibrated')

def _sequence_lengths(query_lengths: list[int], profile: str) -> list[int]:
    observed = [240, 509, 712, 779, 933, 654, 933, 407, 437, 370, 717, 796, 323, 715, 702, 363] if profile == 'draft_decode' else [240, 509, 713, 780, 934, 656, 935, 407, 439, 371, 717, 796, 324, 718, 705, 366]
    indices = torch.linspace(0, 15, len(query_lengths)).round().long().tolist()
    lengths = [observed[index] for index in indices]
    if profile == 'main_prefill':
        requests = len(query_lengths)
        if requests == 1:
            return query_lengths.copy()
        lengths[-1] = min(65532, query_lengths[-1] * (17 - requests))
    return [max(length, queries) for (length, queries) in zip(lengths, query_lengths)]

def _selected_indices(position: int) -> torch.Tensor:
    complete = (position + 1) // 4 * 4
    base_count = min(complete, 2048)
    if complete <= 2048:
        blocks = torch.arange(base_count // 4, dtype=torch.int32, device='cpu')
    else:
        recent_fraction = max(0.1, min(0.25, 0.264 - 1.3e-05 * position))
        block_count = base_count // 4
        recent_count = min(128, round(block_count * recent_fraction))
        older_available = complete // 4 - 128
        older_count = min(block_count - recent_count, older_available)
        recent_count = block_count - older_count
        older = torch.linspace(0, older_available - 1, older_count).round().to(torch.int32)
        recent = torch.linspace(complete // 4 - 128, complete // 4 - 1, recent_count).round().to(torch.int32)
        blocks = torch.cat((older, recent))
    generator = torch.Generator(device='cpu').manual_seed(2017 + position * 37)
    blocks = blocks[torch.randperm(len(blocks), generator=generator, device='cpu')]
    base = (blocks[:, None] * 4 + torch.arange(4, dtype=torch.int32, device='cpu')).reshape(-1)
    tail = torch.arange(complete, position + 1, dtype=torch.int32, device='cpu')
    return torch.cat((base, tail))

def _make_inputs(query_tokens: int, requests: int, profile: str, dtype: str='bfloat16', device: str='cuda', seed: int | None=None) -> dict[str, object]:
    if type(query_tokens) is not int or type(requests) is not int:
        raise TypeError('CSA query and request dimensions must be integers')
    if not 1 <= query_tokens <= MAX_QUERY_TOKENS or not 1 <= requests <= 16:
        raise ValueError('CSA geometry is outside the observed C16 ABI')
    if dtype != 'bfloat16':
        raise ValueError('CSA dtype differs from the observed C16 ABI')
    query_lengths = _query_lengths(query_tokens, requests, profile)
    seq_lens_host = _sequence_lengths(query_lengths, profile)
    generator = torch.Generator(device='cpu').manual_seed(torch.initial_seed() % 2 ** 63 if seed is None else seed)
    query = _profile_tensor((query_tokens, QUERY_HEADS, HEAD_DIM), 'query', profile, generator)
    key = _profile_tensor((query_tokens, 1, HEAD_DIM), 'key', profile, generator)
    values = _profile_tensor((query_tokens, 1, HEAD_DIM), 'value', profile, generator)
    query = query.to(dtype=torch.bfloat16, device=device)
    key = key.to(dtype=torch.bfloat16, device=device)
    value_storage = torch.zeros(query_tokens * 6656, dtype=torch.bfloat16, device=device)
    value = torch.as_strided(value_storage, size=(query_tokens, 1, HEAD_DIM), stride=(6656, HEAD_DIM, 1), storage_offset=6400)
    value.copy_(values.to(dtype=torch.bfloat16, device=device))
    output_std = 0.84 if profile == 'draft_decode' else 0.465 if profile == 'main_prefill' else 0.598
    output = (torch.randn((query_tokens, QUERY_HEADS, HEAD_DIM), generator=generator, device='cpu', dtype=torch.float32) * output_std).to(dtype=torch.bfloat16, device=device)
    cache_storage = torch.zeros((CACHE_BLOCKS, 2, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16, device=device)
    kv_cache = cache_storage.permute(1, 0, 2, 3, 4)
    block_table = torch.zeros((requests, TABLE_WIDTH), dtype=torch.int32, device=device)
    physical_pages: list[list[int]] = []
    next_page = 180
    for (request, length) in enumerate(seq_lens_host):
        count = math.ceil(length / BLOCK_SIZE)
        if next_page + count > CACHE_BLOCKS:
            raise ValueError('source-valid CSA page allocation exceeded the cache')
        pages = list(range(next_page, next_page + count))
        physical_pages.append(pages)
        block_table[request, :count] = torch.tensor(pages, dtype=torch.int32, device=device)
        next_page += count
    req_ids_host: list[int] = []
    positions_host: list[int] = []
    slot_ids_host: list[int] = []
    for (request, (length, queries)) in enumerate(zip(seq_lens_host, query_lengths)):
        prior = length - queries
        if prior:
            positions = torch.arange(prior, dtype=torch.long, device=device)
            pages = block_table[request, positions // BLOCK_SIZE].long()
            offsets = positions.remainder(BLOCK_SIZE)
            cache_profile = 'draft_decode' if profile == 'draft_decode' else 'main_decode'
            prior_keys = _profile_tensor((prior, 1, HEAD_DIM), 'key', cache_profile, generator)
            prior_values = _profile_tensor((prior, 1, HEAD_DIM), 'value', cache_profile, generator)
            kv_cache[0, pages, offsets, 0] = prior_keys[:, 0].to(dtype=torch.bfloat16, device=device)
            kv_cache[1, pages, offsets, 0] = prior_values[:, 0].to(dtype=torch.bfloat16, device=device)
        for position in range(prior, length):
            req_ids_host.append(request)
            positions_host.append(position)
            page = physical_pages[request][position // BLOCK_SIZE]
            slot_ids_host.append(page * BLOCK_SIZE + position % BLOCK_SIZE)
    if len(slot_ids_host) != query_tokens:
        raise ValueError('CSA slot mapping does not conserve query tokens')
    slot_storage = torch.zeros(MAX_QUERY_TOKENS, dtype=torch.int64, device=device)
    slot_storage[:query_tokens] = torch.tensor(slot_ids_host, dtype=torch.int64, device=device)
    slot_mapping = slot_storage[:query_tokens]
    req_storage = torch.zeros(MAX_QUERY_TOKENS, dtype=torch.int32, device=device)
    req_storage[:query_tokens] = torch.tensor(req_ids_host, dtype=torch.int32, device=device)
    req_id_per_token = req_storage[:query_tokens]
    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device=device)
    topk_host = torch.full((MAX_QUERY_TOKENS, INDEX_WIDTH), -1, dtype=torch.int32, device='cpu')
    for (row, position) in enumerate(positions_host):
        selected = _selected_indices(position)
        if selected.numel() > INDEX_WIDTH or (selected.numel() and int(selected[-1]) >= seq_lens_host[req_ids_host[row]]):
            raise ValueError('CSA selected token controls are out of range')
        topk_host[row, :selected.numel()] = selected
    topk_indices_buffer = topk_host.to(device=device)
    return {'query': query, 'key': key, 'value': value, 'kv_cache': kv_cache, 'output': output, 'topk_indices_buffer': topk_indices_buffer, 'metadata_present': True, 'num_actual_tokens': query_tokens, 'skip_write_kvcache': False, 'slot_mapping': slot_mapping, 'block_table': block_table, 'req_id_per_token': req_id_per_token, 'seq_lens': seq_lens, 'num_reqs': requests, 'output_scale': None, 'output_block_scale': None}
