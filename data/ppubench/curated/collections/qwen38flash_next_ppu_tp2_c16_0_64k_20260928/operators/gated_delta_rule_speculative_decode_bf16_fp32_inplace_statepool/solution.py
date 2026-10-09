"""Independent adapter to the pinned FLA speculative decode implementation."""
from torch import nn
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule

class Model(nn.Module):
    def forward(self,q,k,v,g,beta,initial_state,inplace_final_state,cu_seqlens=None,ssm_state_indices=None,num_accepted_tokens=None,retrieve_parent_token=None,use_qk_l2norm_in_kernel=True):
        return fused_recurrent_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,initial_state=initial_state,inplace_final_state=inplace_final_state,cu_seqlens=cu_seqlens,ssm_state_indices=ssm_state_indices,num_accepted_tokens=num_accepted_tokens,retrieve_parent_token=retrieve_parent_token,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
