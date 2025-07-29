import os
import torch

class Config():
    vocab_size = 10000
    context_length = 256
    d_model = 512
    d_ff = 1344
    num_heads = 16
    num_layers = 4
    theta = 10000
    # device = torch.device('mps')
    # device = torch.device('cpu')
    device = torch.device('cuda')
    dtype = torch.float32

    batch_size = 64
    norm_limit = 1
    lr_max = 1e-3
    lr_min = 1e-5
    total_train_iter_cnt = 5000
    train_iter_cnt = 500
    test_iter_cnt = 100
    epoch_cnt = total_train_iter_cnt // train_iter_cnt
    warmup_iter_cnt = int(total_train_iter_cnt * 0.05)
    weight_decay = 0.01
    save_fold = 'models'

