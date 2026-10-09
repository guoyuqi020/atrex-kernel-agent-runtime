from __future__ import annotations
import hashlib,inspect
import torch
from torch import nn
from pathlib import Path

def check_source(callable,digest):
    source=Path(inspect.getsourcefile(callable))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise RuntimeError('independent source version drift')

def norm_receiver(weight,eps,group_size=None):
    from vllm.model_executor.models.qwen4_exp import Qwen4ExpRMSNorm
    check_source(Qwen4ExpRMSNorm,QWEN_SOURCE_SHA)
    receiver=Qwen4ExpRMSNorm.__new__(Qwen4ExpRMSNorm);nn.Module.__init__(receiver)
    receiver.eps=eps;receiver.group_size=group_size;receiver.use_gemma_rms_norm=True;receiver.pre_affine=False;receiver.gated_layernorm=False
    receiver.register_parameter('weight',nn.Parameter(weight,requires_grad=False))
    return receiver

QWEN_SOURCE_SHA='5815e5f3ee9df506fcfc6ea92eee702f70e3a802f0e6fe2b5c805f623905a0ba'
HC_SOURCE_SHA='7fcc977ce8d5fc5444b77b7e2c210c21c32e740197117b2ef6fa75f6bf4688f4'
CONV_SOURCE_SHA='815b4db17be081fca5159ed444abea259195ae9d5f027c8ec074745f5c9259ed'
class Model(nn.Module):
    def __init__(self,hc_count=4,hidden_size=2560,norm_eps=1e-6,per_branch_norm=False):super().__init__();self.hc=hc_count;self.hs=hidden_size;self.eps=norm_eps;self.per_branch=per_branch_norm
    def forward(self,hyper_input,norm_weight,weight_down,weight_up,weight_inject):
        from vllm.model_executor.layers.hyperconnection import _mix_baseline_fn
        check_source(_mix_baseline_fn,HC_SOURCE_SHA)
        receiver=norm_receiver(norm_weight,self.eps,self.hs if self.per_branch else None)
        def call(x,down,up,inject):
            return _mix_baseline_fn(x,receiver,down,up,inject,self.hc,self.hs,self.per_branch)
        mixed,iw=torch.compile(call)(hyper_input,weight_down,weight_up,weight_inject)
        return mixed,hyper_input,iw
