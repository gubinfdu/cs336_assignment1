import os
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn

from .linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        if not d_ff:
            self.d_ff = int(round(d_model * 8 / 3 / 64) * 64)
        else:
            self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        store = self.w1(x)
        x_silu = store * torch.sigmoid(store)
        store2 = x_silu * self.w3(x)
        output = self.w2(store2)
        return output