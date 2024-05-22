import yaml
import random
import numpy as np
import torch

# load the basic config_yaml
def load_config(path):
    with open(path,'r') as stream:
        src_config=yaml.safe_load(stream)
    return src_config

# init the random seed
def init_seed(seed=0,cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic: # search the best CNN algorithm before training
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    else: #default algorithm
        torch.backends.cudnn.deterministic=False
        torch.backends.cudnn.benchmark=True
        

