import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# model 
from model.igev_stereo import IGEVStereo
# utils 
from utils import load_config, init_seed

# get the parameters in commend line
def get_args():
    parser=argparse.ArgumentParser()
    # basic config
    parser.add_argument("--config", type=str, default="configs/igev_kitti.yaml", help="path of the model config file")
    parser.add_argument("--mode",  default="train", choices=["train","test"], help="choose train model or test model")
    parser.add_argument("--logdir", action="store_true", default="./logdir", help="path of the logdir " )
    # about DDP
    parser.add_argument("--master_addr",type=str, default="localhost", help="the master address of the DDP")
    parser.add_argument("--master_port", type=str, default="12355", help="the master port of the DDP ")
    parser.add_argument("--no_distribute", action="store_true", default=False, help="unenable the DDP")
    parser.add_argument("--device", type=str, default="cuda", help="device can be used ")
    parser.add_argument("--restore_hint", type=str, default=0, help="the path of the checkpoints for loading")
    args=parser.parse_args()
    return args

# init DDP
def DDP_init(rank, world_size, master_addr, master_port):
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# main program for mp
def main_worker(rank,world_size,args,configs):
    # initing the DDP setting on GPU : rank  
    is_dist =not args.no_distribute 
    if is_dist:
        if args.master_addr is not None:
            os.environ["MASTER_ADDR"]=args.master_addr
        if args.master_port is not None:
            os.environ["MASTER_PORT"]=args.master_port
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device=torch.device(f"cuda:{rank}")
    else:
        device=torch.device(args.device)
    
    # initing the tensorboard 
    
    
    # init random seed
    seed =0 if args.no_distribute else dist.get_rank()
    print(f"result of GPU{rank} : get_rank(): {dist.get_rank()}")
    init_seed(seed)
    data_cfg=configs["data_cfg"]
    model_cfg=configs["model_cfg"]
    trainer_cfg=configs["trainer_cfg"]
    mode=args.mode 
    model=IGEVStereo(model_cfg)
    print("load model succesfully !")
    



if __name__ == '__main__':
    args=get_args()# basic option
    configs=load_config(args.config) # detail config
    is_dist=not args.no_distribute
    if is_dist:
        print("-Distributed mode")
        world_size=torch.cuda.device_count() 
        print(f"-number of avaliable GPU :{world_size}")
        mp.spawn(main_worker,args=(world_size,args,configs),nprocs=world_size) # for mp ,do not need to set rank, just need to incoming the work_size (equal to num of GPU)
    else:
        print("-Non-distributed mode")

    
