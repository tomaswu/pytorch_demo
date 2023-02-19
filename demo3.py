# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/17 13:14:20
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''
import torch as th
import torch.nn as nn


a=th.arange(16).view(1,4,4).float()


con = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
con2 = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,stride=1,padding=1)

c=con(a)
c2=con2(c)
print(a.shape,c.shape,c2.shape)