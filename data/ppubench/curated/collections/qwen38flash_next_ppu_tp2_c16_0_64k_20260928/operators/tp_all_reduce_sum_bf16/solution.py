"""Zero-conversion production TP collective callable for the paired-rank gate."""

import torch
from torch import nn
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x, inplace=False)
