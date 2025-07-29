import os
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn


def cross_entropy(inputs, targets):
    '''
    inputs: (batch_dim, vocab_size), targets: (batch_dim, )
    '''
    # part1 = inputs - torch.max(inputs, dim=-1, keepdim=True)[0]
    # part2 = torch.logsumexp(part1, dim=-1, keepdim=True)
    # log_score = -(part1 - part2)

    log_sum_exp = torch.logsumexp(inputs, dim=-1, keepdim=True)
    log_score = -(inputs - log_sum_exp)
    
    inds = torch.unsqueeze(targets, -1)
    sel_score = torch.gather(log_score, -1, inds)
    output = sel_score.mean()
    return output


