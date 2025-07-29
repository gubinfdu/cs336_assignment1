import os
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        weight = torch.ones(d_model)
        self.weight = nn.Parameter(weight)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        d_model = self.d_model
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x_norm = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        output = x_norm * self.weight
        return output.to(input_dtype)

