# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/16 08:41:32
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import torch as th
import torch.utils.data as tudata
from PIL import Image,ImageFile
import numpy as np
import os,json
import cv2
import platform

from yolo.datas import *

coco_path = '/Users/tomaswu/programming/coco2017' if 'macOS' in platform.platform() else r'e:\coco2014'
coco_version = 2017 if 'macOS' in platform.platform() else 2014
cf='train'

def padTo(img:np.ndarray,size):
    if len(img.shape)==3:
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,w=img.shape[:2]
    nw,nh=size
    assert nh>=h and nw>=w
    new_image = np.zeros([nh,nw],'uint8')
    new_image[:h,:w]=img
    return new_image

def collate(batch):
    h_max = max([i[1].shape[0] for i in batch])
    w_max = max([i[1].shape[1] for i in batch])
    arr = np.array([padTo(i[1],[w_max,h_max]) for i in batch])
    return arr

coco = COCO(coco_path,cf,coco_version)
dataloader = tudata.DataLoader(coco,50,shuffle=True,num_workers=4,drop_last=False,collate_fn=collate)
if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    import tqdm
    bar=tqdm.trange(len(coco))
    m=0
    for data in dataloader:
        imgs = data
        for i in imgs:
            imshow(i,wait=10)
            bar.update()
    