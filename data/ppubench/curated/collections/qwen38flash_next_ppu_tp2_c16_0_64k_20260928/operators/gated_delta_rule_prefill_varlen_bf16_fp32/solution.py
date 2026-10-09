"""Independent adapter to the pinned serving FLA prefill implementation."""
import torch
from torch import nn
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule

class Model(nn.Module):
    def forward(self,q,k,v,g,beta,initial_state,output_final_state,cu_seqlens=None,head_first=False,use_qk_l2norm_in_kernel=True):
        output=chunk_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,initial_state=initial_state,output_final_state=output_final_state,cu_seqlens=cu_seqlens,head_first=head_first,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        return output[0],output[1],None
