import os
import torch

def softmax(d, dim):
    d -= torch.max(d, dim=dim, keepdim=True)[0]
    d_exp = torch.exp(d)
    output = d_exp / d_exp.sum(dim=dim, keepdim=True)
    return output

