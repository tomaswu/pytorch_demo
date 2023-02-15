# -*- encoding: utf-8 -*-
'''
    @File    :   train.py
    @Time    :   2022/09/12 18:43:40
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import torch as th
from torch import nn
import torchvision
from torch.utils import data
from torch.utils import tensorboard


train_dataset = torchvision.datasets.CIFAR10('./CIFAR10',train=True,transform=torchvision.transforms.ToTensor(),download=False)
test_dataset = torchvision.datasets.CIFAR10('./CIFAR10',train=False,transform=torchvision.transforms.ToTensor(),download=False)

print(f'train dataset length:{len(train_dataset)}\ntest dataset length:{len(test_dataset)}')


train_dataLoader = data.DataLoader(train_dataset,256*4)
test_dataLoader = data.DataLoader(test_dataset,256*4)

class Tomas(nn.Module):
    def __init__(self):
        super().__init__()
        self.module=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.module(x)


if __name__=='__main__':
    import shutil
    cpu = th.device('cpu')
    gpu = th.device("mps")
    try:
        shutil.rmtree('./logs')
    except:
        pass
    model = Tomas().to(gpu)
    loss_func = nn.CrossEntropyLoss().to(gpu)
    optmizer = th.optim.Adam(model.parameters(),lr=5e-3)
    board = tensorboard.SummaryWriter('./logs')

    total_train_step = 0
    total_test_step = 0

    epoch = 20

    for i in range(epoch):
        model.train()
        print(f'\n第{i}轮训练.....')
        for x in train_dataLoader:
            imgs,targets = x
            imgs=imgs.to(gpu)
            targets=targets.to(gpu)
            y_pred = model(imgs)
            loss = loss_func(y_pred,targets)
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
            total_train_step+=1
            board.add_scalar('trains loss',loss,total_train_step)
            print(f'\rtotal train step {total_train_step} : loss: {loss.item()}',end='')

        model.eval() #切换为评估模式，在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭
        test_total_loss = 0
        accurate_count = 0
        with th.no_grad():
            for x in test_dataLoader:
                imgs,targets = x
                imgs=imgs.to(gpu)
                targets=targets.to(gpu)
                y_pred = model(imgs)
                loss=loss_func(y_pred,targets)
                test_total_loss+=loss.item()
                accurate_count+=(y_pred.argmax(1)==targets).sum()
        accurate = accurate_count.item()/len(test_dataset)
        print(f'\n{i} test total loss:{test_total_loss} accurate:{accurate:.2f}')
        board.add_scalar("test loss",test_total_loss,i)
        board.add_scalar("test accurate",accurate,i)
