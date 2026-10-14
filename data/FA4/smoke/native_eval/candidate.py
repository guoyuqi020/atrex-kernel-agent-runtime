"""Exercise R0's supported P128 path through native Eval (not the target P64 ABI)."""
import torch
from vendor.flash_attention.flash_attn.cute import flash_attn_varlen_func


class Model(torch.nn.Module):
    def forward(self, q, k, v, cu_q, page_table):
        output = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=None,
            max_seqlen_q=128, max_seqlen_k=128, page_table=page_table,
            softmax_scale=256**-0.5, causal=True, num_splits=1, pack_gqa=False,
        )
        return output[0] if isinstance(output, (tuple, list)) else output
