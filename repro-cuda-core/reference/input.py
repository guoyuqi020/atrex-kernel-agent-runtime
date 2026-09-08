from __future__ import annotations

import torch


def _ints(value: list[int] | str) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item]
    return [int(item) for item in value]


def _scheduler_metadata(
    *,
    enabled: bool,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_size: int,
) -> torch.Tensor | None:
    if not enabled:
        return None
    try:
        from vllm.attention.utils.fa_utils import get_scheduler_metadata
    except ImportError:
        return None
    return get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        head_dim,
        seqused_k,
        qkv_dtype=torch.bfloat16,
        cu_seqlens_q=cu_seqlens_q,
        page_size=block_size,
        causal=True,
        window_size=(-1, -1),
        num_splits=0,
    )


def _make_inputs(
    q: list[int],
    k: list[int],
    v: list[int],
    query_start_loc: list[int],
    seq_lens: list[int],
    num_kv_heads: int,
    block_table: list[int],
    use_scheduler_metadata: bool,
) -> dict[str, torch.Tensor | None]:
    device = "cuda"
    total_tokens, num_q_heads, head_dim = [int(value) for value in q]
    num_cache_blocks, block_size, kv_heads, kv_head_dim = [int(value) for value in k]
    query_start_loc = _ints(query_start_loc)
    seq_lens = _ints(seq_lens)
    if v != k:
        raise ValueError(f"k/v cache shapes differ: {k} vs {v}")
    if int(num_kv_heads) != kv_heads:
        raise ValueError(f"num_kv_heads={num_kv_heads} but k has {kv_heads}")
    if kv_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: {kv_head_dim} vs {head_dim}")
    if query_start_loc[-1] > total_tokens:
        raise ValueError("active query tokens exceed the CUDA Graph q slot count")

    q_tensor = (
        torch.randn(total_tokens, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    k_cache = (
        torch.randn(
            num_cache_blocks,
            block_size,
            kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    )
    v_cache = (
        torch.randn(
            num_cache_blocks,
            block_size,
            kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    )
    cu_seqlens_q = torch.tensor(query_start_loc, dtype=torch.int32, device=device)
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    max_blocks_per_seq = int(block_table[1])
    block_table_tensor = torch.zeros(
        len(seq_lens),
        max_blocks_per_seq,
        dtype=torch.int32,
        device=device,
    )
    cursor = 0
    for request_id, seq_len in enumerate(seq_lens):
        pages = (int(seq_len) + block_size - 1) // block_size
        block_table_tensor[request_id, :pages] = torch.arange(
            cursor,
            cursor + pages,
            dtype=torch.int32,
            device=device,
        )
        cursor += pages
    if cursor > num_cache_blocks:
        raise ValueError("compact cache does not cover block_table working set")

    q_descale = torch.ones(len(seq_lens), kv_heads, dtype=torch.float32, device=device)
    k_descale = torch.ones_like(q_descale)
    v_descale = torch.ones_like(q_descale)
    scheduler_metadata = _scheduler_metadata(
        enabled=bool(use_scheduler_metadata),
        batch_size=len(seq_lens),
        max_seqlen_q=max(
            int(query_start_loc[index + 1]) - int(query_start_loc[index])
            for index in range(len(seq_lens))
        ),
        max_seqlen_k=max(int(value) for value in seq_lens),
        num_heads_q=num_q_heads,
        num_heads_kv=kv_heads,
        head_dim=head_dim,
        seqused_k=seqused_k,
        cu_seqlens_q=cu_seqlens_q,
        block_size=block_size,
    )
    return {
        "q": q_tensor.contiguous(),
        "k_cache": k_cache.contiguous(),
        "v_cache": v_cache.contiguous(),
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "block_table": block_table_tensor,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "scheduler_metadata": scheduler_metadata,
    }
