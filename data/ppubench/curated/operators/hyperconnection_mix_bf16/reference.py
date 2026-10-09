"""BASELINE HC mix; returns the original residual by alias."""
from __future__ import annotations
import math
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self,hc_count:int=4,hidden_size:int=2560,norm_eps:float=1e-6,per_branch_norm:bool=False):
        super().__init__()
        if hc_count<1 or hidden_size<1 or not math.isfinite(norm_eps) or norm_eps<=0:raise ValueError('invalid HC configuration')
        self.hc=hc_count;self.hs=hidden_size;self.eps=norm_eps;self.per_branch=per_branch_norm
    def forward(self,hyper_input:torch.Tensor,norm_weight:torch.Tensor,weight_down:torch.Tensor,weight_up:torch.Tensor,weight_inject:torch.Tensor):
        hc,hs=self.hc,self.hs
        if hyper_input.ndim!=2 or hyper_input.shape[1]!=hc*hs or hyper_input.dtype!=torch.bfloat16:raise ValueError('expected [T,HC*HS] BF16 input')
        expected_norm=hc*hs if self.per_branch else hs
        if norm_weight.shape!=(expected_norm,) or weight_down.shape[1]!=hc*hs or weight_up.shape!=(hc*hs,weight_down.shape[0]) or weight_inject.shape!=(hc,hc*hs):raise ValueError('HC weight geometry mismatch')
        if any(w.device!=hyper_input.device for w in (norm_weight,weight_down,weight_up,weight_inject)):raise ValueError('HC inputs must share device')
        h=hyper_input.float().reshape(-1,hc,hs)
        w=norm_weight.float().reshape(1,hc if self.per_branch else 1,hs)
        normalized=(h*torch.rsqrt(h.square().mean(-1,keepdim=True)+self.eps)*(1+w)).flatten(1).to(hyper_input.dtype)
        def linear(x,w):return (x.float()@w.float().T).to(hyper_input.dtype)
        gate=torch.nn.functional.silu(linear(normalized,weight_down)/hc)
        gate=torch.sigmoid(linear(gate,weight_up).float()).reshape(-1,hc,hs)
        mixed=(gate*normalized.float().reshape(-1,hc,hs)).mean(-2).to(hyper_input.dtype)
        iw=(2*torch.sigmoid(linear(normalized,weight_inject).float()/hc)).to(hyper_input.dtype)
        return mixed,hyper_input,iw
