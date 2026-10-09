"""Pure shared-expert child; routed MoE and TP reduction are separate."""
from __future__ import annotations
import torch
from torch import nn
class Model(nn.Module):
    def forward(self,x:torch.Tensor,gate_up_weight:torch.Tensor,down_weight:torch.Tensor,expert_gate_weight:torch.Tensor)->torch.Tensor:
        weights=(gate_up_weight,down_weight,expert_gate_weight)
        if x.ndim!=2 or x.dtype!=torch.bfloat16 or any(w.ndim!=2 or w.dtype!=x.dtype or w.device!=x.device for w in weights):raise ValueError('expected BF16 matrices on one device')
        h=down_weight.shape[1]
        if gate_up_weight.shape!=(2*h,x.shape[1]) or down_weight.shape[0]!=x.shape[1] or expert_gate_weight.shape!=(1,x.shape[1]):raise ValueError('shared expert matrix geometry mismatch')
        gate_up=torch.nn.functional.linear(x,gate_up_weight)
        gate,value=gate_up.chunk(2,-1)
        # Pinned PPU SiluAndMul rounds Silu to BF16 before its BF16 multiply.
        activated=torch.nn.functional.silu(gate.float()).to(x.dtype)*value
        down=torch.nn.functional.linear(activated,down_weight)
        score=torch.nn.functional.linear(x,expert_gate_weight)
        return torch.sigmoid(score)*down
