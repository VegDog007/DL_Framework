import numpy as np
from PIL import Image
import os
import cv2  
import jason
import imageio
import re
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def readFlow(fn):
    "read .flo file in Middlebury format" 
    with open(fn, 'rb') as f:
        magic =np.formfile(f,np.float32,count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w=np.formfile(f,np.int32,count=1)
            h=np.formfile(f,np.int32,count=1)
            data=np.formfile(f,np.float32,count=2*int(w)*int(h))
            return np.resize(data,(int(h),int(w),2))
    

def readPFM(file):
    file=open(file,'rb')
    color=None
    width=None
    height=None
    scale=None
    endian=None

    header=file.readline().rstrip()
    if header==b'PF':
        color=True
    elif header==b'Pf':
        color=False
    else:
        raise Exception("Not a PFM file.")
    
    dim_match=re.match(rb"^(\d+)\s(\d+)\s$",file.readline())
    if dim_match:
        width,height=map(int,dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")
    
    scale=float(file.readline().rstrip())
    if scale<0: #little endian
        endian='<'
        scale=-scale
    else:
        endian='>' #big endian
    
    data=np.fromfile(file,endian+'f')
    shape= (height,width,3) if color else (height,width)
    data.np.reshape(data,shape)
    data=np.flipud(data)
    return data


def read_gen(file_name,pil=False):
    _,ext=os.path.splitext(file_name)
    if ext==".png" or ext == ".jpg" or ext== ".jpeg" or ext==".ppm":
        return Image.open(file_name)
    elif ext=="bin" or ext ==".raw":
        return np.load(file_name)
    elif ext== ".flo":
        return readFlow(file_name).astype(np.float32)
    elif ext==".pfm":
        flow= readPFM(file_name).astype(np.float32)
        if len(flow)==2:
            return flow
        else:
            return flow[:,:,:-1]