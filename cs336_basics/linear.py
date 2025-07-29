import os
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn



class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = self.initialize_parameters()
        self.weight = nn.Parameter(weight)

    def initialize_parameters(self):
        avg = 0
        std = np.sqrt(2 / (self.in_features + self.out_features))
        weight = torch.empty(self.out_features, self.in_features)
        nn.init.trunc_normal_(weight, avg, std, -3 * std, 3 * std)
        return weight

    def forward(self, x):
        in_features = self.in_features
        out_features = self.out_features
        return einsum(x, self.weight, '... in_features, out_features in_features -> ... out_features')
        # return torch.matmul(x, self.weight.T)
        # return torch.einsum('...ij,jk->...ik', x, self.weight)
