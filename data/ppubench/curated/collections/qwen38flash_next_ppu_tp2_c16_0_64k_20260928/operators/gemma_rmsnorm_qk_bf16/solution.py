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
    def __init__(self,eps=1e-6):super().__init__();self.eps=eps
    def forward(self,x,weight):return norm_receiver(weight,self.eps)(x)
