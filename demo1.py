# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/15 15:58:07
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    牛刀杀鸡，回归一个三次方程
'''

import torch as th
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm

if th.has_mps:
    device=th.device('mps')
elif th.cuda.is_available():
    device=th.device('cuda')
else:
    device =th.device('cpu')

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
    nn.Linear(1,10),
    nn.Tanh(),
    nn.Linear(10,20),
    nn.Tanh(),
    nn.Linear(20,1)
)

#自定义loss_fun
class CustomLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x,y):
        loss = th.sqrt(th.mean((x-y)**2))
        return loss


# loss_fn = nn.MSELoss()

loss_fn = CustomLoss()

opt=optim.Adam(params=net.parameters(),lr=1e-2)

#这是一个学习率计划，动态调节学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)

with tqdm.trange(10000,colour='#1055aa') as t:
    for epoch in t:
        opt.zero_grad()
        nx=x.reshape(1,50,1)
        nx.to(device)
        yp=net(nx)
        yt=y.reshape(1,50,1)
        yt.to(device)
        loss=loss_fn(yp,yt)
        scheduler.step(loss)
        loss.backward()
        #梯度截断
        th.nn.utils.clip_grad.clip_grad_norm(net.parameters(),100)
        opt.step()

        if epoch%50==0:
            t.set_description_str(f'loss = {th.sum(loss):.3e}')
            ny = net(x.reshape(1,50,1))
            lines[0].set_data(x.detach().numpy(),ny.detach().numpy())
            plt.pause(0.02)

print('train finished!')
plt.ioff()
plt.show()   

