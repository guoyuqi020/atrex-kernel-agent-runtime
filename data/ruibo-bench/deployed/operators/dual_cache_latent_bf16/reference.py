"""Independent V4 sparse MLA core: BF16 stored Q/KV, FP32 math, BF16 out.

Matches the pinned FlashInfer DSv4 sparse callable, not a full attention layer.
RoPE, top-k selection, cache publication and output projection are outside.
"""
from typing import Literal, Optional, Union
import torch

class Model(torch.nn.Module):

    def forward(self, query: torch.Tensor, swa_kv_cache: torch.Tensor, workspace_buffer: torch.Tensor, sparse_indices: torch.Tensor, compressed_kv_cache: Optional[torch.Tensor]=None, sparse_topk_lens: Optional[torch.Tensor]=None, seq_lens: Optional[torch.Tensor]=None, out: Optional[torch.Tensor]=None, bmm1_scale: Union[float, torch.Tensor]=1.0, bmm2_scale: Union[float, torch.Tensor]=1.0, sinks: Optional[torch.Tensor]=None, kv_layout: Literal['HND', 'NHD']='HND', cum_seq_lens_q: Optional[torch.Tensor]=None, max_q_len: Optional[int]=None, enable_pdl: bool | None=None, swa_topk_lens: Optional[torch.Tensor]=None, extra_sparse_indices: Optional[torch.Tensor]=None, extra_sparse_topk_lens: Optional[torch.Tensor]=None) -> torch.Tensor:
        assert query.dtype == swa_kv_cache.dtype == compressed_kv_cache.dtype == torch.bfloat16
        assert query.shape[-1] == 512 and query.shape[-2] in (8, 16, 32, 64, 128)
        assert swa_topk_lens is extra_sparse_indices is extra_sparse_topk_lens is None
        assert isinstance(bmm1_scale, float) and isinstance(bmm2_scale, float)
        assert kv_layout == 'HND' and swa_kv_cache.ndim == compressed_kv_cache.ndim == 3
        assert swa_kv_cache.is_contiguous() and compressed_kv_cache.is_contiguous()
        assert sparse_indices.dtype == sparse_topk_lens.dtype == seq_lens.dtype == torch.int32
        assert query.ndim == 3 and cum_seq_lens_q is not None
        offsets = cum_seq_lens_q.tolist()
        assert cum_seq_lens_q.dtype == torch.int32 and offsets[0] == 0 and (offsets[-1] == len(query))
        q_lengths = [b - a for a, b in zip(offsets, offsets[1:])]
        assert min(q_lengths) > 0 and max_q_len == max(q_lengths)
        assert len(seq_lens) == len(q_lengths) and sparse_indices.shape[0] == len(query)
        assert 128 <= sparse_indices.shape[1] <= 8320 and sparse_indices.shape[1] % 4 == 0
        assert sparse_topk_lens.shape == (len(query),)
        if out is None:
            out = torch.empty_like(query)
        assert out.shape == query.shape and out.dtype == torch.bfloat16
        swa = swa_kv_cache.reshape(-1, 512)
        global_kv = compressed_kv_cache.reshape(-1, 512)
        sink = None if sinks is None else sinks.float()
        if sink is not None:
            assert sinks.dtype == torch.float32 and sinks.shape == (query.shape[1],)
        for request, (first, last) in enumerate(zip(offsets, offsets[1:])):
            final_len = int(seq_lens[request])
            assert final_len >= last - first
            for token in range(first, last):
                visible = final_len - (last - first) + token - first + 1
                swa_count = min(visible, 128)
                total = int(sparse_topk_lens[token])
                assert 128 <= total <= sparse_indices.shape[1]
                i_swa = sparse_indices[token, :swa_count].long()
                i_global = sparse_indices[token, 128:total].long()
                i_swa = i_swa[i_swa >= 0]
                i_global = i_global[i_global >= 0]
                assert bool((i_swa < len(swa)).all()) and bool((i_global < len(global_kv)).all())
                values = torch.cat((swa[i_swa].float(), global_kv[i_global].float()))
                if len(values) == 0:
                    out[token].zero_()
                    continue
                scores = query[token].float() @ values.T * bmm1_scale
                maximum = scores.amax(-1)
                if sink is not None:
                    maximum = torch.maximum(maximum, sink)
                p = (scores - maximum[:, None]).exp()
                denominator = p.sum(-1)
                if sink is not None:
                    denominator = denominator + (sink - maximum).exp()
                out[token].copy_((p @ values * (bmm2_scale / denominator[:, None])).to(out.dtype))
        return out
