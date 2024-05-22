import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from PIL import Image

class BaseTrainer:
    def __init__(self,
                 model: nn.Module=None,
                 trainer_cfg: dict=None,
                 data_cfg: dict=None,
                 is_dist: bool= True,
                 rank: int=None,
                 device: torch.device = torch.device("cpu"),
                 **kwargs
                 ):
        # self.msg=...
        self.model= model
        self.trainer_cfg= trainer_cfg
        self.data_cfg= data_cfg
        self.optimizer_cfg=trainer_cfg["optimizer_cfg"]
        self.scheduler_cfg=trainer_cfg["scheduler_cfg"]
        self.evaluator_cfg=trainer_cfg["evaluator_cfg"]
        self.optimizer=None