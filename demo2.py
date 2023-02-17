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

coco = COCO(coco_path,cf,coco_version)
dataloader = tudata.DataLoader(coco,10,shuffle=True,num_workers=0,drop_last=False)

for data in dataloader:
    print(data)
    break