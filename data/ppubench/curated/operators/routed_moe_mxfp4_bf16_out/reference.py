"""Semantic TP-only routed MXFP4 expert reference for the PPU source branch."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


FP4_TABLE = (0., .5, 1., 1.5, 2., 3., 4., 6.,
             0., -.5, -1., -1.5, -2., -3., -4., -6.)


def _dequant(weight: torch.Tensor, scale: torch.Tensor, expert: int) -> torch.Tensor:
    packed = weight[expert].transpose(0, 1)
    packed_scales = scale[expert].transpose(0, 1)
    table = torch.tensor(FP4_TABLE, dtype=torch.float32, device=packed.device)
    raw = packed.to(torch.int64)
    values = table[torch.stack((raw & 15, (raw >> 4) & 15), dim=-1).flatten(-2)]
    exponent = packed_scales.to(torch.int32) - 127
    factors = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    factors = factors.repeat_interleave(32, dim=-1)
    if values.shape != factors.shape:
        raise ValueError("runtime MXFP4 weight and scale layouts disagree")
    return (values * factors).to(torch.bfloat16)


class Model(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        w13_weight_packed: torch.Tensor,
        w13_scale_packed: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_weight_packed: torch.Tensor,
        w2_scale_packed: torch.Tensor,
        w2_bias: torch.Tensor,
        global_num_experts: int,
        top_k: int,
        activation: str,
    ) -> torch.Tensor:
        if activation != "silu" or global_num_experts != router_logits.shape[1]:
            raise ValueError("MoE activation or expert count differs from the PPU source")
        if x.shape[0] != router_logits.shape[0] or top_k != 10:
            raise ValueError("MoE token count or routing fanout differs")
        probabilities = torch.softmax(router_logits.float(), dim=-1)
        weights, ids = torch.topk(probabilities, top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        flat_ids = ids.reshape(-1).long()
        flat_weights = weights.to(torch.bfloat16).reshape(-1)
        routed = torch.zeros((flat_ids.numel(), x.shape[1]),
                             dtype=x.dtype, device=x.device)
        for expert in torch.unique(flat_ids, sorted=True).tolist():
            mask = flat_ids == expert
            token_ids = torch.nonzero(mask, as_tuple=False).flatten() // top_k
            w13 = _dequant(w13_weight_packed, w13_scale_packed, expert)
            fc1 = F.linear(x[token_ids], w13)
            fc1 = fc1 + w13_bias[expert].to(fc1.dtype)
            intermediate = fc1.shape[-1] // 2
            activated = (
                F.silu(fc1[:, :intermediate].float())
                * fc1[:, intermediate:].float()
            ).to(fc1.dtype)
            w2 = _dequant(w2_weight_packed, w2_scale_packed, expert)
            fc2 = F.linear(activated, w2)
            fc2 = fc2 + w2_bias[expert].to(fc2.dtype)
            routed[mask] = (
                fc2.float() * flat_weights[mask, None].float()
            ).to(x.dtype)
        return routed.view(x.shape[0], top_k, x.shape[1]).sum(dim=1, dtype=torch.float32).to(x.dtype)
