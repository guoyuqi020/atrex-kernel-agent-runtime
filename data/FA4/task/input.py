from __future__ import annotations

import torch


def _make_inputs(
    query_start_loc_values: list[int],
    seq_lens_values: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    block_table_columns: int,
    workspace_size_bytes: int,
    bmm1_scale: float,
    bmm2_scale: float,
) -> dict[str, object]:
    device = "cuda"
    query_start_loc = query_start_loc_values
    seq_lens = seq_lens_values
    if len(query_start_loc) != len(seq_lens) + 1:
        raise ValueError("query_start_loc and seq_lens disagree")
    query_lengths = [
        end - start
        for start, end in zip(query_start_loc, query_start_loc[1:])
    ]
    if any(length <= 0 for length in query_lengths):
        raise ValueError("prefill query lengths must be positive")
    if any(query > total for query, total in zip(query_lengths, seq_lens)):
        raise ValueError("query length cannot exceed KV sequence length")

    pages_per_request = [
        (length + page_size - 1) // page_size for length in seq_lens
    ]
    total_pages = sum(pages_per_request)
    if max(pages_per_request) > block_table_columns:
        raise ValueError("block table capacity is too small")

    query = (
        torch.randn(
            query_start_loc[-1],
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 1.22
    ).to(torch.float8_e4m3fn)
    kv_cache = (
        torch.randn(
            total_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.96
    ).to(torch.float8_e4m3fn)

    block_tables = torch.full(
        (len(seq_lens), block_table_columns),
        -1,
        dtype=torch.int32,
        device=device,
    )
    page_cursor = 0
    cumulative_pages = [0]
    for request_index, page_count in enumerate(pages_per_request):
        block_tables[request_index, :page_count] = torch.arange(
            page_cursor,
            page_cursor + page_count,
            dtype=torch.int32,
            device=device,
        )
        page_cursor += page_count
        cumulative_pages.append(page_cursor)

    return {
        "query": query,
        "kv_cache": kv_cache,
        "workspace_buffer": torch.zeros(
            workspace_size_bytes,
            dtype=torch.uint8,
            device=device,
        ),
        "block_tables": block_tables,
        "seq_lens": torch.tensor(
            seq_lens, dtype=torch.int32, device=device
        ),
        "max_q_len": max(query_lengths),
        "max_kv_len": max(seq_lens),
        "bmm1_scale": bmm1_scale,
        "bmm2_scale": bmm2_scale,
        "batch_size": len(seq_lens),
        "cum_seq_lens_q": torch.tensor(
            query_start_loc, dtype=torch.int32, device=device
        ),
        "cum_seq_lens_kv": torch.tensor(
            cumulative_pages, dtype=torch.int32, device=device
        ),
        "window_left": -1,
        "sinks": None,
        "o_sf_scale": None,
        "out": torch.empty(
            query_start_loc[-1],
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
    }
