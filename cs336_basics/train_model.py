import os
import math
import tqdm
from typing import BinaryIO
import numpy as np
from einops import rearrange, einsum
import torch
from torch import nn

from .transformer import Transformer
from .loss_func import cross_entropy
from .optimizer import AdamW, grad_clip, lr_scheduler
from .utils import softmax
from .config import Config

def get_batch(d, batch_size, context_length, device):
    starts = np.random.randint(0, len(d) - context_length, batch_size)
    x = np.vstack([d[i: i + context_length] for i in starts])
    x = torch.tensor(x)
    y = np.vstack([d[i + 1: i + 1 + context_length] for i in starts])
    y = torch.tensor(y)
    return (x.to(device), y.to(device))

def get_test_iter(d, batch_size, context_length, device):
    batch_cnt = (len(d) - context_length - 1) // batch_size
    for i in range(batch_cnt):
        starts = range(i * batch_size, (i + 1) * batch_size)
        x = np.vstack([d[i: i + context_length] for i in starts])
        x = torch.tensor(x)
        y = np.vstack([d[i + 1: i + 1 + context_length] for i in starts])
        y = torch.tensor(y)
        yield (x.to(device), y.to(device))

def save_checkpoint(model, optimizer, iteration, out):
    store_data = {}
    store_data['model'] = model.state_dict()
    store_data['optimizer'] = optimizer.state_dict()
    store_data['iteration'] = iteration
    torch.save(store_data, out)

def load_checkpoint(source, model, optimizer):
    store_data = torch.load(source)
    model.load_state_dict(store_data['model'])
    optimizer.load_state_dict(store_data['optimizer'])
    return store_data['iteration']

def create_model(vocab_size, context_length, d_model, d_ff, num_heads, num_layers, theta, device=None, dtype=None):
    model = Transformer(vocab_size, context_length, d_model, d_ff, num_heads, num_layers, theta, device=device, dtype=dtype)
    return model

def load_parameter(model, parameter_path):
    pass

def train(model, train_data, test_data, config):
    logs = []
    optimizer = AdamW(model.parameters(), config.lr_max, config.weight_decay)
    best_loss = np.inf

    for i in range(config.epoch_cnt):
        model.train()
        train_loss = 0
        for j in tqdm.tqdm(range(config.train_iter_cnt)):
        # for j in range(config.train_iter_cnt):
            x, y = get_batch(train_data, config.batch_size, config.context_length, config.device)
            x = x.to(torch.int)
            y = y.to(torch.long)
            logit = model(x)
            loss = cross_entropy(logit.reshape(-1, logit.shape[-1]), y.reshape(-1, ))
            optimizer.zero_grad()
            loss.backward()
            grad_clip(model.parameters(), config.norm_limit)
    
            lr = lr_scheduler(i * config.train_iter_cnt + j, config.lr_max, config.lr_min, config.warmup_iter_cnt, config.total_train_iter_cnt)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / config.train_iter_cnt

        model.eval()
        test_loss = 0
        for _ in tqdm.tqdm(range(config.test_iter_cnt)):
        # for _ in range(config.test_iter_cnt):
            with torch.no_grad():
                x, y = get_batch(test_data, config.batch_size, config.context_length, config.device)
                x = x.to(torch.int)
                y = y.to(torch.long)
                logit = model(x)
                loss = cross_entropy(logit.reshape(-1, logit.shape[-1]), y.reshape(-1, ))
            test_loss += loss.item()
        test_loss = test_loss / config.test_iter_cnt
        print ('epoch: {0}/{1}, train_loss: {2}, test_loss: {3}'.format(i + 1, config.epoch_cnt, train_loss, test_loss))
        msg = [i + 1, train_loss, test_loss]
        logs.append(msg)
        # print(f'iter {i + 1:05d}: valid loss = {test_loss:.4f}')
        
        if (test_loss < best_loss) and (i >= 0):
            path = config.save_fold + '/model_{0}.pt'.format(i + 1)
            save_checkpoint(model, optimizer, i + 1, path)
            best_loss = test_loss
            print ('saved')
    return logs

def top_p_sampling(logit, top_p):
    '''
    logit: (vocab_size, )
    p: float, 0-1
    '''
    score = softmax(logit, dim=-1)
    sort_score, sort_ind = torch.sort(score, dim=-1, descending=True)
    cum_score = torch.cumsum(sort_score, dim=-1)
    cut_ind = torch.argmax((cum_score >= top_p).to(int), dim=-1) + 1
    sel_ind = sort_ind[: cut_ind]
    sel_score = sort_score[: cut_ind]
    sel_score /= sel_score.sum()
    return sel_ind, sel_score
    

def generate_token(model, x, max_gen_token_cnt, eos_token_id, temperature, top_p=0.9):
    '''
    x: torch.tensor, (1, seq_len) or (seq_len, )
    '''
    if x.dim() == 1:
        x = x[None, ]
    orig_seq_len = x.shape[1]
    with torch.no_grad():
        for i in range(max_gen_token_cnt):
            output = model(x) # (batch_size, seq_len, vocab_size)
            logit = output[:, -1] # (batch_size, vocab_size)
            logit /= temperature
            sel_ind, sel_score = top_p_sampling(logit, top_p)
            token_id = sel_ind[torch.multinomial(sel_score, 1)]
            x = torch.concat([x, torch.tensor([[token_id]])], dim=-1)
            if token_id == eos_token_id:
                break
    return x[:, orig_seq_len: ]






if __name__ == '__main__':
    config = Config()
    model = create_model(
        config.vocab_size, config.context_length, config.d_model, config.d_ff, config.num_heads, config.num_layers, config.theta, 
        device=config.device, dtype=config.dtype
    )

    

