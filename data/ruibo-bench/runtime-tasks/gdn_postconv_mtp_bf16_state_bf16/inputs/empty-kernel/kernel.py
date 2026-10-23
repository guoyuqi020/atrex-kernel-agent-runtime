import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Fresh CUDA bootstrap: implement the operator from scratch")
