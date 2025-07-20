import os
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        k = torch.tensor(range(0, d_k, 2))
        theta = torch.arange(max_seq_len)[:, None] / torch.pow(self.theta, k / d_k) # (max_seq_len, d_k // 2)
        theta = torch.repeat_interleave(theta, 2).reshape(max_seq_len, -1) # (max_seq_len, d_k)
        self.cos = torch.cos(theta).to(device)
        self.sin = torch.sin(theta).to(device)
        self.register_buffer('cos_table', self.cos, persistent=False)
        self.register_buffer('sin_table', self.sin, persistent=False)
        self.device = device

        # self.r = torch.zeros((max_seq_len, d_k, d_k))
        # for i in range(max_seq_len):
        #     self.r[i] = self.generate_rotate_matrix(i)
    
    # def generate_rotate_matrix(self, token_ind):
    #     r = torch.zeros(self.d_k, self.d_k)
    #     for k in range(0, self.d_k // 2, 1):
    #         theta = token_ind / (self.theta ** (2 * k / self.d_k))
    #         theta = torch.tensor(theta)
    #         r[2 * k: 2 * k + 2, 2 * k: 2 * k + 2] = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
    #     return r

    def forward(self, x, token_positions=None):
        '''
        x = range(8) -> x2 = [-1, 0, -3, 2, -5, 4, -7, 6]
        '''
        *batch_size, seq_len, d_k = x.shape
        inds = torch.flip(torch.arange(d_k).reshape(-1, 2), dims=[-1]).reshape(-1, ).to(torch.int).to(self.device)
        signs = torch.ones(d_k).reshape(-1, 2)
        signs[:, 0] *= -1
        signs = signs.reshape(-1, ).to(torch.int).to(self.device)
        x2 = torch.index_select(x, -1, inds) * signs

        # x_reshape = x.reshape(-1, seq_len, d_k) # (batch, seq_len, d_k)
        # part_even = torch.unsqueeze(x_reshape[:, :, : : 2], -1) # (batch, seq_len, d_k // 2, 1)
        # part_odd = torch.unsqueeze(x_reshape[:, :, 1: : 2], -1) # (batch, seq_len, d_k // 2, 1)
        # x2 = torch.concat([-part_odd, part_even], dim=-1) # (batch, seq_len, d_k // 2, 2)
        # x2 = x2.reshape(x.shape) # (batch, seq_len, d_k)

        if token_positions is None:
            token_positions = torch.arange(seq_len).to(torch.int).to(self.device)
        output = x * self.cos[token_positions] + x2 * self.sin[token_positions] # (batch, seq_len, d_k)
        return output
    
    # def rope(self, x, token_ind):
    #     x1 = x
    #     part1 = x[: : 2][None, ]
    #     part2 = -x[1: : 2][None, ]
    #     x2 = torch.concat([part2, part1]).T.reshape(-1, )
    #     x_rotate = x1 * self.cos[token_ind] + x2 * self.sin[token_ind]
    #     return x_rotate
    
    # def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    #     print (x.shape, token_positions.shape)
    #     # np.save('rope_input.npz', x)
    #     *batch_size, seq_len, d_k = x.shape
    #     output = torch.zeros_like(x)
    #     for i in range(batch_size):
    #         for j in range(seq_len):
    #             token_ind = token_positions[j]
    #             token_emb = x[i, j]
    #             output[i, j] = self.rope(token_emb, token_ind)
    #             # output[i, j] = torch.matmul(self.generate_rotate_matrix(token_ind), token_emb)
    #     # np.save('rope_output.npy', output)
    #     return output

    
        

# if __name__ == '__main__':
#     rope = RotaryPositionalEmbedding(10000, 64, 12)
#     # print (rope.generate_rotate_matrix(1))
