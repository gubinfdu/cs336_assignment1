import os
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn

from .linear import Linear
from .rope import RotaryPositionalEmbedding

def softmax(d, dim):
    d -= torch.max(d, dim=dim, keepdim=True)[0]
    d_exp = torch.exp(d)
    output = d_exp / d_exp.sum(dim=dim, keepdim=True)
    return output

def scaled_dot_product_attention(Q, K, V, mask=None):
    '''
    query, key: (..., seq_len, d_k)
    value: (..., seq_len, d_v)
    '''
    d_k = Q.shape[-1]
    logit = einsum(Q, K, '... seq_len_query d_k, ... seq_len_key d_k -> ... seq_len_query seq_len_key') / torch.sqrt(torch.tensor(d_k))
    if mask is not None:
        logit -= (1 - mask.to(torch.int)) * 1e10
    score = softmax(logit, dim=-1)
    output = einsum(score, V, '... seq_len_query seq_len_key, ... seq_len_key d_v -> ... seq_len_query d_v')
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False, theta=None, max_seq_len=None, token_position=None, device=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
            self.token_position = token_position
        else:
            self.rope = None
        self.device = device
    
    def split_multi_head(self, x):
        shape = x.shape
        *batch_dim, seq_len, d_model = shape
        x_split_head = x.reshape(batch_dim + [seq_len, self.num_heads, self.d_k]).transpose(-3, -2) # (batch_dim, num_head, seq_len, d_k)
        return x_split_head
    
    def merge_multi_head(self, x):
        shape = x.shape
        *batch_dim, num_head, seq_len, d_k = shape
        x_merge = x.transpose(-3, -2).reshape(batch_dim + [seq_len, num_head * d_k]) # (batch_dim, seq_len, d_model)
        return x_merge
    
    def forward(self, x):
        '''
        input -> qkv linear -> split to multi-head -> rope -> attention with mask -> merge multi-head -> output linear
        x: (..., seq_len, d_model)
        '''
        shape = x.shape
        *batch_dim, seq_len, d_model = shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        qs = self.split_multi_head(Q)
        ks = self.split_multi_head(K)
        vs = self.split_multi_head(V)
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.to(bool).to(self.device)
        if self.use_rope:
            qs_rope = self.rope(qs, self.token_position)
            ks_rope = self.rope(ks, self.token_position)
            x2 = scaled_dot_product_attention(qs_rope, ks_rope, vs, mask) # (batch_dim, num_head, seq_len, d_k)
        else:
            x2 = scaled_dot_product_attention(qs, ks, vs, mask) # (batch_dim, num_head, seq_len, d_k)
        x2 = self.merge_multi_head(x2) # (batch_dim, seq_len, d_model)
        output = self.o_proj(x2)
        return output



