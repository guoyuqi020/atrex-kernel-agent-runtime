"""Zero-conversion production TP collective callable for the paired-rank gate."""

import torch
from torch import nn
from vllm.distributed.communication_op import tensor_model_parallel_all_gather


class Model(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim not in (0, -1):
            raise ValueError("unsupported AllGather dimension")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_gather(x, dim=self.dim)
