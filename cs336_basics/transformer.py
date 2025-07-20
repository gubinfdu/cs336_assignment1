import os
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn

from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .feed_forward_net import SwiGLU
from .attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, token_position=None, device=None, dtype=None):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, True, theta, max_seq_len, token_position, device=device)
        self.device = device
    
    def forward(self, x):
        res = self.norm1(x)
        res = self.attn(res)
        x2 = x + res

        res = self.norm2(x2)
        res = self.ffn(res)
        output = x2 + res
        return output

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, d_ff, num_heads, num_layers, theta, device=None, dtype=None):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = context_length
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.theta = theta

        self.emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, self.max_seq_len, theta, device=device, dtype=dtype) for i in range(num_layers)
        )
        self.norm_final = RMSNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.dense_final = Linear(d_model, vocab_size)
    
    def forward(self, inds):
        x = self.emb(inds)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        output = self.dense_final(x)
        return output


if __name__ == '__main__':
    vocab_size = 10000
    context_length = 16
    d_model = 64
    d_ff = 128
    num_heads = 4
    num_layers = 3
    theta = 10000
    # model = Transformer(vocab_size, context_length, d_model, d_ff, num_heads, num_layers, theta)
    def get_params_cnt(vocab_size, max_seq_len, num_layer, num_head, d_model, d_ff, ffn_type='v1'):
        emb_cnt = vocab_size * d_model
        norm_cnt = d_model
        linear_final_cnt = d_model * vocab_size
        
    #     linear_cnt = d_model * d_ff
        rope_cnt = max_seq_len * d_model * 2
        attn_cnt = [d_model * d_model * 4, rope_cnt]
        attn_cnt = [d_model * d_model * 4]
        if ffn_type == 'v1':
            ffn_cnt = [d_model * d_ff * 2]
        else:
            ffn_cnt = [d_model * d_ff * 3]
        block_cnt = [norm_cnt * 2] + ffn_cnt + attn_cnt
        
        total_cnt = [emb_cnt, norm_cnt, linear_final_cnt]
        for i in range(num_layer):
            total_cnt += block_cnt
        return total_cnt
    vocab_size = 50257
    max_seq_len = 1024
    d_model = 1600
    d_ff = 6400
    num_head = 25
    num_layer = 48
    cnt = get_params_cnt(vocab_size, max_seq_len, num_layer, num_head, d_model, d_ff)
    sum(cnt) # 1635537600

