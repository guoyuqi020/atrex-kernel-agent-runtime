"""Qwen4Exp Q/K Gemma RMS normalization, without residual or gates."""
from __future__ import annotations
import math
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self,eps:float=1e-6):
        super().__init__()
        if not math.isfinite(eps) or eps<=0:raise ValueError('eps must be positive and finite')
        self.eps=eps
    def forward(self,x:torch.Tensor,weight:torch.Tensor)->torch.Tensor:
        if x.ndim!=3 or x.shape[-1]!=256 or x.dtype!=torch.bfloat16 or weight.shape!=(256,) or weight.device!=x.device:
            raise ValueError('expected BF16 [T,H,256] and weight [256] on one device')
        value=x.float();normalized=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+self.eps)
        return (normalized*(1+weight.float())).to(x.dtype)
