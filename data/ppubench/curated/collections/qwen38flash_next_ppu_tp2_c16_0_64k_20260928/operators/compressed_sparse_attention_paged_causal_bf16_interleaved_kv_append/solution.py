"""Private replay adapter for the unmodified CSA backend source call."""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.v1.attention.backends.qwen4_exp_csa import Qwen4ExpCSAFlashAttentionImpl


SOURCE_SHA256 = "0598d1ebe2206fcfb5dbefd2df60a9b242f934c5741f656e67de6ec9aef3f94e"


class Model(nn.Module):
    def __init__(
        self,
        scale: float,
        kv_cache_dtype: str,
        layer_k_scale: torch.Tensor,
        layer_v_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        source = Path(inspect.getsourcefile(Qwen4ExpCSAFlashAttentionImpl.forward) or "")
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError("unmodified CSA backend source SHA-256 differs")
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        self.register_buffer("layer_k_scale", torch.as_tensor(layer_k_scale, dtype=torch.float32))
        self.register_buffer("layer_v_scale", torch.as_tensor(layer_v_scale, dtype=torch.float32))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        topk_indices_buffer: torch.Tensor,
        metadata_present: bool,
        num_actual_tokens: int,
        skip_write_kvcache: bool,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        req_id_per_token: torch.Tensor,
        seq_lens: torch.Tensor,
        num_reqs: int,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        os.environ["VLLM_QWEN4_EXP_CSA_REFERENCE"] = "1"
        layer = SimpleNamespace(
            csa_indexer=SimpleNamespace(topk_indices_buffer=topk_indices_buffer),
            _k_scale=self.layer_k_scale,
            _v_scale=self.layer_v_scale,
        )
        metadata = (
            SimpleNamespace(
                num_actual_tokens=num_actual_tokens,
                skip_write_kvcache=skip_write_kvcache,
                slot_mapping=slot_mapping,
                block_table=block_table,
                req_id_per_token=req_id_per_token,
                seq_lens=seq_lens,
                num_reqs=num_reqs,
            )
            if metadata_present
            else None
        )
        return Qwen4ExpCSAFlashAttentionImpl.forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            metadata,
            output,
            output_scale,
            output_block_scale,
        )
