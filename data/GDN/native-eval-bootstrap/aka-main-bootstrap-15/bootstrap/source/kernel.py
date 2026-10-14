"""Fixed evaluator adapter for the official non-CP FlashInfer GDN fallback."""

import torch
import torch.nn as nn

from flashinfer.gdn_kernels.blackwell.gdn_prefill import (
    chunk_gated_delta_rule_sm100,
)


class Model(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output: torch.Tensor,
        output_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        chunk_gated_delta_rule_sm100(
            q=q,
            k=k,
            v=v,
            gate=g,
            beta=beta,
            output=output,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            output_state=output_state,
            scale=q.shape[-1] ** -0.5,
        )
        return {"output": output, "final_state": output_state}
