# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/15 15:58:07
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    牛刀杀鸡，回归一个三元方程
'''

import torch as th
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np

#数据准备
x=th.linspace(0,10,50).reshape(50)
y=0.1*(x-5)**3+0.12*x**2-x+3# + 3*(th.rand_like(x)-0.5)

plt.ion()
fig,ax = plt.subplots()
ax.scatter(x,y)
lines = ax.plot([],[],color = 'r')

plt.show()

#网略构建
net = nn.Sequential(
    nn.Linear(1,100),
    nn.Tanh(),
    nn.Linear(100,200),
    nn.Tanh(),
    nn.Linear(200,1)
)
loss_fn = nn.MSELoss()

opt=optim.SGD(params=net.parameters(),lr=1e-2)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)


for epoch in range(100000):
    opt.zero_grad()
    yp=net(x.reshape(1,50,1))
    loss=loss_fn(yp,y.reshape(1,50,1))
    scheduler.step(loss)
    loss.backward()
    #梯度截断
    th.nn.utils.clip_grad.clip_grad_norm(net.parameters(),100)
    opt.step()
    
    
    if epoch%50==0:
        print(f'epoch = {epoch}  loss = {th.sum(loss)}')
        ny = net(x.reshape(1,50,1))
        lines[0].set_data(x.detach().numpy(),ny.detach().numpy())
        plt.pause(0.02)

print('train finished!')
plt.ioff()
plt.show()   

