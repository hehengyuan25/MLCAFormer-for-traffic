import json
import os
import random

import numpy as np
import torch


class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()
        return (data - self.mean) / self.std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        device, dtype = data.device, data.dtype
        data1 = (data * torch.tensor(self.std, device=device, dtype=dtype)) + \
                torch.tensor(self.mean, device=device, dtype=dtype)
        return data1

def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            with open(log, "a") as f:
                print(*values, file=f, end=end)
        else: # it is a file object
            print(*values, file=log, end=end)
            log.flush()

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)

def vrange(starts, stops):
    stops = np.asarray(stops)
    l = stops - starts
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])