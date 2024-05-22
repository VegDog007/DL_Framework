import torch
import torch.nn.functional as F

from modeling.base_model import BaseModel
from utils import Odict
from types import SimpleNamespace

from .igev_stereo import IGEVStereo

import torchvision.transforms.functional as TF
from PIL import Image
def issue_transform(img,type=1):# input tensor
    #print("image in:",img.shape)
    h=img.shape[2]
    w=img.shape[3]

    if type==1: #convert to grayscale
        img=TF.rgb_to_grayscale(img)
        img=torch.cat((img,img,img),dim=1)
        
    elif type==2: # downsample and convert to gray 
        
        img=TF.rgb_to_grayscale(img)
        img=TF.resize(img,(h//2,w//2),interpolation=Image.BILINEAR)
        img=TF.resize(img,(h,w),interpolation=Image.BILINEAR)
        img=torch.cat((img,img,img),dim=1)
    elif type==3: # downsample
        img=TF.resize(img,(h//2,w//2),interpolation=Image.BILINEAR)
        img=TF.resize(img,(h,w),interpolation=Image.BILINEAR)
    elif type==4:
        
        img=TF.rgb_to_grayscale(img)
        kernel=torch.tensor(([0,0,0,0],
                             [0,0,0,0],
                             [0,1,0,0],
                             [0,0,0,0]),dtype=torch.float32).cuda()
        kernel= kernel.unsqueeze(0).unsqueeze(0)

        img=F.conv2d(img.unsqueeze(0),kernel,bias=None, stride=4)
        
        img=TF.resize(img,(h,w),interpolation=Image.BILINEAR)
        img=torch.cat((img,img,img),dim=1)
    #print("image out :",img.shape)
    return img



class NewLoss:
    def __init__(self, loss_gamma=0.9, max_disp=192):
        super().__init__()
        self.loss_gamma = loss_gamma
        self.max_disp = max_disp

    def __call__(self, training_output):
        training_disp = training_output['disp']
        pred_disp = training_disp['disp_ests']
        disp_gt = training_disp['disp_gt']
        mask = training_disp['mask']
        
        disp_init_pred, disp_preds_GRU1, disp_preds_GRU2 = pred_disp

        loss = self.sequence_loss(disp_preds_GRU1, disp_preds_GRU2, disp_init_pred, disp_gt, mask.float())

        loss_info = Odict()
        loss_info['scalar/train/loss_disp'] = loss.item()
        loss_info['scalar/train/loss_sum'] = loss.item()
        return loss, loss_info
    @staticmethod
    
    def sequence_loss(disp_preds_GRU1, disp_preds_GRU2, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    #""" Loss function defined over sequence of flow predictions """
        n_predictions = len(disp_preds_GRU1)
        assert n_predictions==len(disp_preds_GRU2)
        assert n_predictions >= 1
        

        disp_loss = 0.0
        disp_gt = disp_gt.unsqueeze(1)
        mag = torch.sum(disp_gt**2, dim=1).sqrt()
        #print("1",mag.shape)

        #print("3",valid.shape)
        valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()


        disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], size_average=True)
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss1 = (disp_preds_GRU1[i] - disp_gt).abs()
            i_loss2 = (disp_preds_GRU2[i] - disp_gt).abs()
            assert i_loss1.shape == valid.shape, [i_loss1.shape, valid.shape, disp_gt.shape, disp_preds_GRU2[i].shape]
            assert i_loss2.shape == valid.shape, [i_loss2.shape, valid.shape, disp_gt.shape, disp_preds_GRU2[i].shape]
            disp_loss += i_weight * i_loss1[valid.bool()].mean()
            disp_loss +=  i_weight * i_loss2[valid.bool()].mean()

        return disp_loss

class NewStereo(BaseModel):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)


    def build_network(self):
        model_cfg = self.model_cfg
        self.net = IGEVStereo(model_cfg['base_config'])

    def init_parameters(self):
        return

    def build_loss_fn(self):
        """Build the loss."""
        self.loss_fn = NewLoss()

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"] #image 1
        tgt_img = inputs["tgt_img"] #image 2
        img3=tgt_img #img3=img2
        tgt_img=ref_img #img2=img1

        #ref_img=issue_transform(ref_img,2)
        #tgt_img=issue_transform(tgt_img,2)


        res = self.net(ref_img, tgt_img, img3)
        if self.training:
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": res,
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/train/disp_c': torch.cat([inputs['disp_gt'][0], res[0].squeeze(1)[0]], dim=0),
                },
            }
        else:
            disp_pred = res.squeeze(1)
            output = {
                "inference_disp": {
                    "disp_est": disp_pred,
                },
                "visual_summary": {
                    'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/test/disp_c': disp_pred[0],
                }
            }
            if 'disp_gt' in inputs:
                output['visual_summary'] = {
                    'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/val/disp_c': torch.cat([inputs['disp_gt'][0], disp_pred[0]], dim=0),
                }
        return output
