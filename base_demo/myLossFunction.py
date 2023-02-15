# -*- encoding: utf-8 -*-
'''
    @File    :   myLossFunction.py
    @Time    :   2022/08/13 10:21:47
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''



import torch as th
import torch.nn as nn

# loss: 计算实际输出和预期输出的误差

x = th.tensor([1,2,3]).reshape([1,1,1,3]).type(th.float32)
y = th.tensor([1,2,5]).reshape([1,1,1,3]).type(th.float32)


loss = nn.L1Loss()

print(loss(x,y))

print(nn.MSELoss()(x,y))

closs = nn.CrossEntropyLoss()

x = th.tensor([0.1,0.2,0.3]).reshape([1,3])
y = th.tensor([1])

print(closs(x,y))