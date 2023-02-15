# -*- encoding: utf-8 -*-
'''
    @File    :   module_demo.py
    @Time    :   2022/08/13 10:00:51
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import torch as th
import torch.nn as nn



class TModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 =  nn.Conv2d(3,32,5,1,2)  # in_channels,out_channels,kernel_size,stride,padding
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,32,5,1,2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,5,1,2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*4*4,64)
        self.linear2 = nn.Linear(64,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        return x

module = TModule()
print(module)

x = th.ones([64,3,32,32])

print(module(x).shape)
