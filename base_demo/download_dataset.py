# -*- encoding: utf-8 -*-
'''
    @File    :   download_dataset.py
    @Time    :   2022/08/11 22:19:47
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


import torchvision

train_set = torchvision.datasets.CIFAR10(root='./CIFAR10',train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False,download=True)

img,label = train_set[0]

print(img)
print(train_set.classes[label])