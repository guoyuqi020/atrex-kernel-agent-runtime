"""Independent float32 attention reference for the upstream P128 smoke."""
import torch


class Model(torch.nn.Module):
    def forward(self, q, k, v, cu_q, page_table):
        query = q.float().transpose(0, 1).unsqueeze(0)
        key = k[0].float().transpose(0, 1).unsqueeze(0).expand(1, 16, 128, 256)
        value = v[0].float().transpose(0, 1).unsqueeze(0).expand(1, 16, 128, 256)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=256**-0.5,
        ).squeeze(0).transpose(0, 1).to(torch.bfloat16)
