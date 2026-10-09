import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, hc_count=4, hidden_size=2560):
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size

    def forward(self, block_output, residual, mix_aux):
        raise NotImplementedError("Fresh CUDA bootstrap: implement the operator from scratch")
