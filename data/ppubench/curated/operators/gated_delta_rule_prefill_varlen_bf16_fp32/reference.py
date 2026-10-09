"""Portable chunk-solve reference for the captured packed GDN prefill ABI.

For each block, solve the lower-triangular delta update system, then apply
query readout and the final state update. Independent sequential FP64 tests
verify this algebra. Q/K normalization preserves the source BF16 roundtrip.
This draft has not yet passed real-input replay or target PPU acceptance.
"""
from __future__ import annotations

import torch
from torch import nn


def _normalize(value: torch.Tensor) -> torch.Tensor:
    floating = value.float()
    return (floating * torch.rsqrt(floating.square().sum(-1, keepdim=True) + 1e-6)).to(value.dtype).float()


class Model(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        head_first: bool = False,
        use_qk_l2norm_in_kernel: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if head_first or not output_final_state or not use_qk_l2norm_in_kernel:
            raise ValueError("only the captured sequence-first normalized final-state ABI is supported")
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != (8, 128) or k.shape != q.shape:
            raise ValueError("q/k must have shape [1,T,8,128]")
        tokens = q.shape[1]
        if not 1 <= tokens <= 8192 or v.shape != (1,tokens,24,128):
            raise ValueError("v or token count differs from the captured ABI")
        if g.shape != (1,tokens,24) or beta.shape != g.shape:
            raise ValueError("g/beta must have shape [1,T,24]")
        for name,value,dtype in (("q",q,torch.bfloat16),("k",k,torch.bfloat16),
                                 ("v",v,torch.bfloat16),("g",g,torch.float32),
                                 ("beta",beta,torch.bfloat16),("initial_state",initial_state,torch.float32)):
            if value.dtype != dtype or value.device != q.device:
                raise ValueError(f"{name} dtype or device differs from the captured ABI")
        if cu_seqlens is None or cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
            raise ValueError("packed prefill requires one-dimensional int32 cu_seqlens")
        bounds = cu_seqlens.tolist()
        if (not 2 <= len(bounds) <= 17 or bounds[0] != 0 or bounds[-1] != tokens
                or any(start > end for start,end in zip(bounds,bounds[1:]))):
            raise ValueError("cu_seqlens must monotonically cover all packed tokens")
        if initial_state.shape != (len(bounds)-1,24,128,128):
            raise ValueError("initial_state must contain one [24,V,K] state per sequence")

        # [H,T,D]. Value heads are grouped three-to-one with key heads.
        queries = _normalize(q)[0].repeat_interleave(3,dim=1).transpose(0,1) * (128**-.5)
        keys = _normalize(k)[0].repeat_interleave(3,dim=1).transpose(0,1)
        values = v[0].transpose(0,1).float()
        gates = g[0].transpose(0,1)
        betas = beta[0].transpose(0,1).float()
        output = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)
        for sequence,(start,end) in enumerate(zip(bounds,bounds[1:])):
            state = initial_state[sequence].clone()
            for begin in range(start,end,64):
                stop = min(begin+64,end)
                length = stop-begin
                query,key,value = queries[:,begin:stop],keys[:,begin:stop],values[:,begin:stop]
                weight = betas[:,begin:stop,None]
                cumulative_gate = gates[:,begin:stop].cumsum(-1)
                prefix = cumulative_gate.exp()
                lower = torch.ones((length,length),dtype=torch.bool,device=q.device).tril()
                difference = cumulative_gate[:,:,None]-cumulative_gate[:,None,:]
                # Mask before exp: irrelevant upper-triangle differences must
                # not overflow and become inf*0 in the triangular products.
                decay = difference.masked_fill(~lower,float('-inf')).exp()
                correlation = key @ key.transpose(-1,-2)
                system = (weight*decay*correlation).tril(-1)
                system = system + torch.eye(length,dtype=torch.float32,device=q.device)
                right = weight*(value-prefix[:,:,None]*(key @ state.transpose(-1,-2)))
                updates = torch.linalg.solve_triangular(system,right,upper=False,unitriangular=True)
                readout = prefix[:,:,None]*(query @ state.transpose(-1,-2))
                readout = readout + ((query @ key.transpose(-1,-2))*decay) @ updates
                output[0,begin:stop] = readout.transpose(0,1).to(v.dtype)
                weighted_keys = key*decay[:,-1,:,None]
                state = prefix[:,-1,None,None]*state + updates.transpose(-1,-2) @ weighted_keys
            final_state[sequence].copy_(state)
        return output,final_state,None
