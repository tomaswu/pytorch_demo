# -*- encoding: utf-8 -*-
'''
    @File    :   conv_test.py
    @Time    :   2022/08/11 23:42:20
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import torch.nn as nn
import torchvision as tv
import torch as th
import torch.utils.tensorboard as tb

import shutil
try:
    shutil.rmtree('./logs')
except:
    pass

dataset = tv.datasets.CIFAR10(root='./CIFAR10',train=False,download=True,transform=tv.transforms.ToTensor())    

dataloader = th.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)

class TModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 每个outchannel对应一层kernel
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=3,stride=1)
        self.relu1 = nn.ReLU(inplace=False) #inplace:如果为True,则会在原数据上进行操作
        self.linear = nn.Linear(in_features=6*28*28,out_features=10,bias=True)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxPool(x)
        x=self.relu1(x)
        x=self.linear(x.flatten(start_dim=1))
        return x

module = TModule()
print(module)

writer = tb.SummaryWriter('./logs')

for step,data in enumerate(dataloader):
    imgs,targets = data
    writer.add_images('test_data',imgs,step)
    y = module(imgs)
    print(step,y.shape)
    # writer.add_images("output",y.reshape(-1,3,28,28),step)