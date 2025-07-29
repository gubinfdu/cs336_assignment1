import os
import pickle
import torch

def load_data(path):
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
            return d
    except:
        print (path)

def softmax(d, dim):
    d -= torch.max(d, dim=dim, keepdim=True)[0]
    d_exp = torch.exp(d)
    output = d_exp / d_exp.sum(dim=dim, keepdim=True)
    return output

