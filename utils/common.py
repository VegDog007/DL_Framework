import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import inspect


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
        

class c():
    def __call__(self, *args, **kwds):
        pass
    def __getattr__(self,*args,**kwargs):
        def no_op(*args,**kwargs): pass
        return no_op
    def __exit__(self,exc_type,exc_value,exc_tb):
        pass
    def __enter__(self):
        pass
    def dampening(self):
        return self
    

class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v


def get_attr_from(sources,name):
    try:
        return getattr(sources[0],name)
    except get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0],name)

def get_valid_args(obj, input_args, free_keys=[]):
    # obj: object  need to be checked
    # input_args: all the args
    # free_keys: the keys that can be ignored
    # return: reture the parameters that can be used in obj function or class
    if inspect.isfunction(obj):
        expected_args = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_args = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError("obj must be function or class")
    
    unexpected_keys=[]
    expected_args={}
    for k,v in input_args.items():
        if k in expected_args:
            expected_args[k]=v
        elif k in free_keys:
            pass
        else:
            unexpected_keys.append(k)
    if unexpected_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpected_keys), obj.__name__))
    return expected_args

def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)