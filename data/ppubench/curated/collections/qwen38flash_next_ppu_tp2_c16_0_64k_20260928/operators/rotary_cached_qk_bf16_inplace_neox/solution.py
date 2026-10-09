"""Independent installed custom op; cache is projected from module state."""
import hashlib,inspect
from pathlib import Path
from torch import nn
class Model(nn.Module):
    def __init__(self,head_size=256,rotary_dim=64,is_neox_style=True):super().__init__();self.head_size=head_size;self.is_neox=is_neox_style
    def forward(self,positions,query,key,cos_sin_cache):
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
        from vllm import _custom_ops as ops
        path=Path(inspect.getsourcefile(RotaryEmbedding))
        if hashlib.sha256(path.read_bytes()).hexdigest()!='906f2ec0fb330f0e8abeda973c004ec1ebfa407f45fec46f73adf07ca376e2d8':raise RuntimeError('rotary source drift')
        ops.rotary_embedding(positions,query,key,self.head_size,cos_sin_cache,self.is_neox)
        return query,key
