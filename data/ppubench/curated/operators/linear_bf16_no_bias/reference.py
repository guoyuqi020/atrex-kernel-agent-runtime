"""Unquantized no-bias BF16 projection; local TP output only."""
from __future__ import annotations
import torch
from torch import nn
class Model(nn.Module):
    def forward(self,x:torch.Tensor,weight:torch.Tensor)->torch.Tensor:
        if x.ndim!=2 or weight.ndim!=2 or x.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16 or x.shape[1]!=weight.shape[1] or x.device!=weight.device:
            raise ValueError('expected compatible BF16 matrices on one device')
        return (x.float()@weight.float().T).to(torch.bfloat16)
