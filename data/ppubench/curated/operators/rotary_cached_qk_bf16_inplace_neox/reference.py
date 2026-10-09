"""Supplied-cache BF16 Neox rotary reference with full Q/K mutation semantics."""
from __future__ import annotations
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self,head_size:int=256,rotary_dim:int=64,is_neox_style:bool=True):
        super().__init__()
        if (head_size,rotary_dim,is_neox_style)!=(256,64,True):raise ValueError('unsupported captured rotary contract')
        self.head_size=head_size;self.rotary_dim=rotary_dim
    def forward(self,positions:torch.Tensor,query:torch.Tensor,key:torch.Tensor,cos_sin_cache:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        if positions.dtype!=torch.int64 or cos_sin_cache.dtype!=torch.bfloat16 or cos_sin_cache.ndim!=2 or cos_sin_cache.shape[1]!=self.rotary_dim:raise ValueError('rotary control/cache contract drift')
        rows=cos_sin_cache.index_select(0,positions.flatten());half=self.rotary_dim//2;cos=rows[:,:half].unsqueeze(1);sin=rows[:,half:].unsqueeze(1)
        for value in (query,key):
            if value.dtype!=torch.bfloat16 or value.ndim!=2 or value.shape[0]!=positions.numel() or value.shape[1]%self.head_size or value.stride(-1)!=1:raise ValueError('rotary head layout drift')
        q_heads=query.unflatten(-1,(-1,self.head_size));k_heads=key.unflatten(-1,(-1,self.head_size))
        def rotated(heads):
            first=heads[...,:half];second=heads[...,half:self.rotary_dim]
            return torch.cat((first*cos-second*sin,second*cos+first*sin),dim=-1).to(torch.bfloat16).flatten(1)
        q_indices=(torch.arange(query.shape[1]//self.head_size,device=query.device)[:,None]*self.head_size+torch.arange(self.rotary_dim,device=query.device)).flatten()
        k_indices=(torch.arange(key.shape[1]//self.head_size,device=key.device)[:,None]*self.head_size+torch.arange(self.rotary_dim,device=key.device)).flatten()
        query[:,q_indices]=rotated(q_heads)
        key[:,k_indices]=rotated(k_heads)
        return query,key
