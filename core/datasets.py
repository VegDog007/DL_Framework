import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import os
import random
from core.utils import frame_utils
import copy
from glob import glob

class MyDataset(Dataset):
    def __init__(self,aug_params=None,sparse=False,reader=None):
        self.augmentor=None
        self.sparse=sparse
        # data argumentation
        self.img_pad=aug_params.pop('img_pad',None) if aug_params is not None else None
        # the disparity reader 
        if reader is None:
            self.disparity_reader=frame_utils.read_gen
        else: 
            self.disparity_reader=reader
        # data augmentation
        if aug_params is not None and "crop_size" in aug_params:    
            if sparse:
                self.augmentor=SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor=FlowAugmentor(**aug_params)
        self.is_test=False
        self.init_seed=False
        self.flow_list=[]
        self.disparity_list=[]
        self.image_list=[]
        self.extra_info=[]
    
    def __getitem__(self,index):
        if self.is_test:
            img1=frame_utils.read_gen(self.image_list[index][0])
            img2=frame_utils.read_gen(self.image_list[index][1])
            img1=np.array(img1).astype(np.uint8)[..., :3]
            img2=np.array(img2).astype(np.uint8)[..., :3]
            img1=torch.from_numpy(img1).permute(2,0,1).float()
            img2=torch.from_numpy(img2).permute(2,0,1).float()
            return img1,img2,self.extra_info[index]
        if not self.init_seed:
            worker_info = torch.uitls.data.get_worker_info()
            if worker_info is not None:
                # set random seed for torch,numpy and random
                seed=worker_info.id
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
        
        index=index%len(self.image_list)
        # get disp and valid 
        disp=self.disparity_reader(self.disparity_list[index])
        if isinstance(disp,tuple):
            disp,valid=disp
        else:
            valid=disp<512

        # get image
        img1=frame_utils.read_gen(self.image_list[index][0])
        img2=frame_utils.read_gen(self.image_list[index][1])
        img1=np.array(img1).astype(np.uint8)
        img2=np.array(img2).astype(np.uint8)
        disp=np.array(disp).astype(np.float32)
        flow=np.stack([disp,np.zoeros_like(disp)],axis=-1)

        #dealing with grayscale images or color images
        if len(img1.shape)==2:
            img1=torch.tile(img1[... ,None], (1,1,3))
            img2=torch.tile(img2[... ,None], (1,1,3))
        else:
            img1=img1[..., :3]
            img2=img2[..., :3]
        
        # dada augmentation
        if self.augmentor is not None:
            if self.sparse:
                img1,img2,flow,valid=self.augmentor(img1,img2,flow,valid)
            else:
                img1,img2,flow=self.augmentor(img1,img2,flow)
        
        img1=torch.from_numpy(img1).permute(2,0,1).float()
        img2=torch.from_numpy(img2).permute(2,0,1).float()
        flow=torch.from_numpy(flow).permute(2,0,1).float()
        if self.sparse:
            valid=torch.from_numpy(valid)
        else:
            valid=(flow[0].abs()<512) & (flow[1].abs()<512 )

        # image padding
        if self.img_pad is not None:
            padW,padH=self.img_pad
            img1=F.pad(img1,[padW]*2+ [padH]*2)
        flow=flow[:1]
        return self.image_list[index]+[self.disparity_list[index]],img1, img2, flow, valid.float()
    def __len__(self):
        return len(self.image_list)
    def __mul__(self,v):
        self_copy=copy.deepcopy(self)
        self_copy.image_list=v* self.image_list
        self_copy.disparity_list=v* self.disparity_list
        self_copy.extra_info=v* self.extra_info
        self_copy.flow_list=v* self.flow_list
        return self_copy


class KITTI12(MyDataset):
    def __init__(self,aug_params=None,root='path/KITTI2012',image_set="training"):
        super(KITTI12,self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)
        image1_list=sorted(glob(os.path.join(root,image_set,"colored_0/*_10.png")))
        image2_list=sorted(glob(os.path.join(root,image_set,"colored_1/*_10.png")))
        disp_list=sorted(glob(os.path.join(root,image_set,"disp_occ/*_10.png"))) if image_set=="training" \
            else [os.path.join(root,"training/disp_occ/000085_10.png")]* len(image1_list)
         
        for idx,(img1,img2,disp) in enumerate(zip(image1_list,image2_list,disp_list)):
            self.image_list+=[[img1,img2]]
            self.disparity_list+=[disp]


class KITTI15(MyDataset):
    def __init__(self, aug_params=None, root="path/KITTI2015", image_set="training"):
        super(KITTI15,self).__init__(aug_params,sparse=True,reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)
        image1_list=sorted(glob(os.path.join(root,image_set,"image_2/*_10.png")))
        image2_list=sorted(glob(os.path.join(root,image_set,"image_3/*_10.png")))
        disp_list=sorted(glob(os.path.join(root,image_set,"disp_occ/*_10.png"))) if image_set=="training" \
            else [os.path.join(root,"training/disp_occ/000085_10.png")]*len(image2_list)
        
        for idx,(img1,img2,disp) in enumerate(zip(image1_list,image2_list,disp_list)):
            self.image_list+=[[img1,img2]]
            self.disparity_list+=[disp]

class SceneFlow(MyDataset):
    def __init__(self, aug_params=None, root="path/SceneFlow",dstype='frames_finalpass',thing_test=False):
        super(SceneFlow,self).__init__(aug_params,sparse=False)
        self.root=root
        self.dstype=dstype
        if thing_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    