# -*- encoding: utf-8 -*-
'''
    @File    :   mySequential.py
    @Time    :   2022/08/13 09:45:11
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import torch as th
import torch.nn as nn
import torch.utils.data as td
import torchvision as tv
import torch.utils.tensorboard as tb
import shutil

try:
    shutil.rmtree('./logs')
except:
    pass


# out shape: (hm+2*padding-2*dilation*(kernel-1))/stride+1


seq = nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(in_features=64*4*4,out_features=64),
    nn.Linear(in_features=64,out_features=10)
)

loss = nn.CrossEntropyLoss()
optimizer = th.optim.Adam(seq.parameters(),lr=0.01)


writer = tb.SummaryWriter(log_dir='logs')

train_data = tv.datasets.CIFAR10('./CIFAR10/',train=True,download=True,transform=tv.transforms.ToTensor())

train_loader = td.DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)


for epoch in range(20):
    running_loss = 0.0
    for data in train_loader:
        imgs,targets=data
        optimizer.zero_grad();
        y = seq(imgs)
        res_loss = loss(y,targets)
        res_loss.backward()
        running_loss += res_loss
        optimizer.step()
    print(running_loss)
    
