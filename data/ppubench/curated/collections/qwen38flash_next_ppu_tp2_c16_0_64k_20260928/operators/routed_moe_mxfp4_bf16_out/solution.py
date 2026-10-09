"""Run the deployed TP-only MXFP4 MoE path from the benchmark tensor ABI."""

from __future__ import annotations

import torch
from torch import nn


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
        from vllm.utils.import_utils import import_triton_kernels

        import_triton_kernels()
        from triton_kernels.matmul_ogs import PrecisionConfig
        from triton_kernels.tensor import FP4, Tensor
        from triton_kernels.tensor_details import layout
        from triton_kernels.tensor_details.layout import StridedLayout
        from vllm.model_executor.layers.fused_moe.config import (
            mxfp4_w4a16_moe_quant_config,
        )
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
        from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
            UnfusedOAITritonExperts,
        )

        if activation != "silu" or top_k != 10 or x.shape[0] != router_logits.shape[0]:
            raise ValueError("MoE activation, top-k or token count differs from deployment")
        if global_num_experts != router_logits.shape[1]:
            raise ValueError("MoE expert count differs from deployment")
        value_layout, _ = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
        scale_layout, _ = layout.make_default_matmul_mxfp4_w_scale_layout(
            mx_axis=1, num_warps=8,
        )
        if value_layout is not StridedLayout or scale_layout is not StridedLayout:
            raise ValueError("benchmark tensors require the PPU strided MXFP4 layout")

        def wrap_weight(packed: torch.Tensor) -> Tensor:
            shape = list(packed.shape)
            shape[1] *= 2
            return Tensor(packed, dtype=FP4, shape=shape)

        w13 = wrap_weight(w13_weight_packed)
        w2 = wrap_weight(w2_weight_packed)
        config = mxfp4_w4a16_moe_quant_config(
            w1_scale=PrecisionConfig(weight_scale=Tensor(w13_scale_packed)),
            w2_scale=PrecisionConfig(weight_scale=Tensor(w2_scale_packed)),
            w1_bias=w13_bias,
            w2_bias=w2_bias,
        )
        weights, ids, _, _, _ = fused_topk(
            hidden_states=x,
            gating_output=router_logits,
            topk=top_k,
            renormalize=True,
            indices_type=torch.int32,
        )
        experts = UnfusedOAITritonExperts(config, split_routing=True)
        _, tokens, intermediate, hidden, routed_top_k = experts.moe_problem_size(
            x, w13, w2, ids,
        )
        workspace13_shape, workspace2_shape, output_shape = experts.workspace_shapes(
            tokens, intermediate, hidden, routed_top_k,
            global_num_experts, global_num_experts, None,
        )
        workspace13 = torch.empty(workspace13_shape, dtype=x.dtype, device=x.device)
        workspace2 = torch.empty(workspace2_shape, dtype=x.dtype, device=x.device)
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        experts.apply(
            output, x, w13, w2, weights, ids, activation,
            global_num_experts, None, None, None,
            workspace13, workspace2, None, False,
        )
        return output
