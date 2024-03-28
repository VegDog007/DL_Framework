import os
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from tools import * 
import core.datasets as datasets
import logging
# assigning the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = accelerator.device # if use hugging face accelerator 
#AMP
try:
    from torch.cuda.amp import GardScaler
except:
    class CardScaler():
        def __init__(self):
            pass
        def scale(self,loss):
            return loss
        def unscale_(self,optimizer):
            pass
        def step(self,optimizer):
            optimizer.step()
        def update(self):
            pass
    
#Create the optimizer and learning rate scheduler
def creat_optimizer(args,model):

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

class Logger:
    
    def __init__(self,model,scheduler):
        self.model=model
        self.scheduler=scheduler
        self.total_steps=0
        self.running_loss={}
        self.writer=SummaryWriter(log_dir=args.logdir)


def train(args):
    model=NewStereo(args)
    print(f"Parameter Count: {count_parameters(model)}")

    train_loader=datasets.fetch_dataloader(args)

    optimizer,scheduler=creat_optimizer(args,model)
    total_steps = 0

    logger=Logger(model,scheduler)

    # check latest model checkpoints
    latest_model=AutoLoad_Checkpoint(args.logdir)

    if latest_model is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.to(device)
    # Enabling the Dropout and BatchNormalization
    model.train()
    
    model.freeze_bn() # Keep BatchNorm forzen , provided in model 

    valid_frequency=10000

    # code here --------------------------------------------------------------gyt


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--logdir',default='./checkpoints/',help="path of checkpoints")
    # training parameters
    parser.add_argument('--batch_size',type=int,default=8,help="batch size of training")
    parser.add_argument('--tain_datasets',args='+',default=['sceneflow'],help='add one or more dataset for training')
    parser.add_argument("--num_steps",type= int ,default=200000 ,help="length of training scheduler")
    parser.add_argument('--crop_imagesize', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")


    args=parser.parse_args()

    train(args)