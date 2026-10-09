import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, hc_count=4, hidden_size=2560):
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size

    def forward(self, block_output, residual, mix_aux):
        shape = (*block_output.shape[:-1], self.hc_count, self.hidden_size)
        return (residual.reshape(shape) + block_output.unsqueeze(-2) * mix_aux.unsqueeze(-1)).flatten(-2)
