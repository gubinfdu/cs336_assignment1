import os
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn

from .linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        if not d_ff:
            self.d_ff = int(round(d_model * 8 / 3 / 64) * 64)
        else:
            self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

        # weight = self.initialize_parameters(self.d_ff, self.d_model)
        # self.w1 = nn.Parameter(weight)
        # weight = self.initialize_parameters(self.d_model, self.d_ff)
        # self.w2 = nn.Parameter(weight)
        # weight = self.initialize_parameters(self.d_ff, self.d_model)
        # self.w3 = nn.Parameter(weight)
    
    # def initialize_parameters(self, dim1, dim2):
    #     avg = 0
    #     std = np.sqrt(2 / (dim1 + dim2))
    #     weight = torch.empty(dim1, dim2, device=self.device, dtype=self.dtype)
    #     nn.init.trunc_normal_(weight, avg, std, -3 * std, 3 * std)
    #     return weight

    # def forward(self, x):
    #     d_model = self.d_model
    #     d_ff = self.d_ff
    #     part1 = einsum(x, self.w1, '... d_model, d_ff d_model -> ... d_ff')
    #     part1 = part1 * torch.sigmoid(part1)
    #     part2 = einsum(x, self.w3, '... d_model, d_ff d_model -> ... d_ff')
    #     x = part1 * part2
    #     output = einsum(x, self.w2, '... d_ff, d_model d_ff -> ... d_model')
    #     return output

    def forward(self, x):
        store = self.w1(x)
        x_silu = store * torch.sigmoid(store)
        store2 = x_silu * self.w3(x)
        output = self.w2(store2)
        return output

