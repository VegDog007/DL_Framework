import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader,BatchSampler,RandomSampler,SqueueBatchSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.common import get_attr_from, get_valid_args, NoOp

from utils.Trainer_common import fix_bn

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
        # cfg
        self.trainer_cfg= trainer_cfg
        self.data_cfg= data_cfg
        # 
        self.optimizer_cfg=trainer_cfg["optimizer_cfg"]
        self.scheduler_cfg=trainer_cfg["scheduler_cfg"]
        self.evaluator_cfg=trainer_cfg["evaluator_cfg"]
        self.optimizer=None
        self.epoch_scheduler=NoOp()
        self.batch_scheduler=NoOp()
        self.warmup_shcheduler=NoOp()
        self.clip_grad=NoOp()
        #DDP setting
        self.is_dist=is_dist
        self.rank=rank if is_dist else None
        self.device=torch.device("cuda",rank) if is_dist else device 
        # record
        self.current_epoch= 0
        self.current_iters= 0
        self.save_path = os.path.join(
            self.trainer_cfg.get("save_path", "./output"),
            self.data_cfg['name'], self.model.model_name, self.trainer_cfg['save_name']
        )
        # AMP
        self.amp = self.trainer_cfg.get("amp", False)
        if self.amp:
            self.scaler = torch.cuda.nn.GradScaler()
        # data
        self.train_loader= None
        self.val_loader= None
        self.test_loader= None
        # building the trainer....
    def build_model(self,*args,**kwargs):
        # apply fix batch norm
        if self.trainer_cfg.get("fix_bn",False):
            #msg ...
            self.model =fix_bn(self.model)
        # init parameters
        if self.trainer_cfg.get("init_parameters",False):
            #msg
            self.model.init_parameters()

    def build_optimizer(self,optimizer_cfg):
        # build optimizer
        optimizer= get_attr_from([optim], optimizer_cfg["solver"])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ["solver"])
        optimizer= optimizer(params =[p for p in self.model.parameters() if p.requires_grad], **valid_arg)
        self.optimizer=optimizer

    def build_schduler(self,scheduler_cfg):
        # msg
        scheduler=get_attr_from(optim.lr_scheduler,scheduler_cfg["scheduler"])
        valid_arg = get_valid_args(scheduler, scheduler_cfg, ["solver","warmup","on_epoch"])
        scheduler=scheduler(self.optimizer, **valid_arg)
        if scheduler_cfg.get("on_epoch",False):
            self.epoch_scheduler=scheduler
        else:
            self.batch_scheduler=scheduler