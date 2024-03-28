import os
import torchvision.transforms.functional as TF
import torch   
import PIL.Image as Image 
# auto check the checkpoint in "logdir" and return the latest model's path
def AutoLoad_Checkpoint(logdir):
    latest_time=0
    latest_model=None
    for filename in os.listdir(logdir):
        if filename.endswith(".pth"):
            model_path=os.path.join(logdir,filename)
            model_time=os.path.getctime()
            if model_time>latest_time:
                latest_time=model_time
                latest_model=model_path
    if latest_model is not None:
        print(f"Load the model: {latest_model}......")
        return latest_model
    else:
        print("No checkpoint exists !")
        return None
    
# Data transforms

def My_transforms(img,type=1):# input tensor
    h=img.shape[2]
    w=img.shape[3]

    if type==1: #convert to grayscale
        img=TF.rgb_to_grayscale(img)
        img=torch.cat((img,img,img),dim=1)        
    elif type==2: # downsample and convert to gray ,and upsample 
        img=TF.rgb_to_grayscale(img)
        img=TF.resize(img,(h//4,w//4),interpolation=Image.BILINEAR)
        img=TF.resize(img,(h,w),interpolation=Image.BILINEAR)
        img=torch.cat((img,img,img),dim=1)
    elif type==3: # downsample and upsample 
        img=TF.resize(img,(h//4,w//4),interpolation=Image.BILINEAR)
        img=TF.resize(img,(h,w),interpolation=Image.BILINEAR)
    elif type==4: # pixelshuffle method to downsample
        
        img=TF.rgb_to_grayscale(img)
        kernel=torch.tensor(([0,0,0,0],
                             [0,0,0,0],
                             [0,1,0,0],
                             [0,0,0,0]),dtype=torch.float32).cuda()
        kernel= kernel.unsqueeze(0).unsqueeze(0)

        img=F.conv2d(img.unsqueeze(0),kernel,bias=None, stride=4)
        
        img=TF.resize(img,(h,w),interpolation=Image.BILINEAR)
        img=torch.cat((img,img,img),dim=1)

    return img

#count model parameters
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if p.requires_grad)
    