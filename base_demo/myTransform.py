# -*- encoding: utf-8 -*-
'''
    @File    :   myTransform.py
    @Time    :   2022/08/11 00:03:57
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

path = './hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(path)

transformer = transforms.ToTensor()
tensor_img = transformer(img)

#输入均值和标准差  (value-mean)/std
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

tensor_img = trans_norm(tensor_img)

print(tensor_img)

#有一个resize函数主要是对image格式进行编辑

# compose 将不同的变换结合起来
trans_compose = transforms.Compose([transformer,trans_norm])

img2 = trans_compose(img)
print(img2)
#RandomCrop随机裁剪
trans_random = transforms.RandomCrop(100)
for i in range(10):
    crop = trans_random(img)
    cv2.imshow('test',cv2.cvtColor(np.array(crop),cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    
 