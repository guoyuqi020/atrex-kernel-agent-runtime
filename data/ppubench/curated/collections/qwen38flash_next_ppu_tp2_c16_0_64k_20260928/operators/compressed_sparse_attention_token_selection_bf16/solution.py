"""Private replay adapter for the unmodified source selector function."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch
import torch.nn as nn

from vllm.model_executor.layers.qwen_compressed_sparse_attn_indexer import (
    select_csa_prefill_tokens_reference,
)


SOURCE_SHA256 = "f309fdb5fecbd00499e1e9eb5b0878a8f1758bade64e92166541ae250a1e1c74"


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        source = Path(inspect.getsourcefile(select_csa_prefill_tokens_reference) or "")
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError("unmodified selector source SHA-256 differs")

    def forward(
        self,
        q: torch.Tensor,
        compressed_keys: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        token_topk: int,
        compress_ratio: int,
    ) -> torch.Tensor:
        return select_csa_prefill_tokens_reference(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            query_positions,
            sequence_lengths,
            token_topk,
            compress_ratio,
        )
