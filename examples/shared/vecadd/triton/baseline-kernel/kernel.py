"""Shared seed copied into a Triton lineage framework-baseline session.

The Core framework-baseline phase replaces this reference-shaped placeholder with
a correct Triton implementation before the Runtime registers the first Kernel.
"""

import torch


class Model(torch.nn.Module):
    def forward(self, left, right):
        return left + right
