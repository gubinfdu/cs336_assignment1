import os
import math
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            print ('error')
        else:
            defaults = {'lr': lr}
            super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        if lr < 0:
            print ('error')
        else:
            defaults = {'lr': lr}
            super().__init__(params, defaults)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lambda_ = weight_decay
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                
                m = state.get('m', torch.zeros_like(grad))
                m = self.beta1 * m + (1 - self.beta1) * grad
                state['m'] = m

                v = state.get('v', torch.zeros_like(grad))
                v = self.beta2 * v + (1 - self.beta2) * grad ** 2
                state['v'] = v

                t = state.get('t', 1)
                alpha_t = lr * math.sqrt(1 - self.beta2 ** t) / (1 - self.beta1 ** t)
                state['t'] = t + 1
                p.data -= alpha_t * m / (torch.sqrt(v) + self.eps) + self.lr * self.lambda_ * p.data
        return loss

def lr_scheduler(t, alpha_max, alpha_min, warmup_iter_cnt, cos_iter_cnt):
    if t < warmup_iter_cnt:
        return t / warmup_iter_cnt * alpha_max
    elif t <= cos_iter_cnt:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos((t - warmup_iter_cnt) / (cos_iter_cnt - warmup_iter_cnt) * np.pi))
    else:
        return alpha_min

def grad_clip(params, norm_limit, eps=1e-6):
    grad = [p.grad.data for p in params if p.grad is not None]
    norm = torch.sqrt(sum([(g ** 2).sum() for g in grad]))
    if norm <= norm_limit:
        pass
    else:
        for p in params:
            if (p.grad is None) or (p.requires_grad == False):
                continue
            p.grad.data *= norm_limit / (norm + eps)


