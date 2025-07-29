import os
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn



class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight = self.initialize_parameters()
        self.weight = nn.Parameter(weight)

    def initialize_parameters(self):
        avg = 0
        std = 1
        weight = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.trunc_normal_(weight, avg, std, -3 * std, 3 * std)
        return weight
    
    def forward(self, token_ids):
        return self.weight[token_ids]

