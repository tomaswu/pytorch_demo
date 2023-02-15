# -*- encoding: utf-8 -*-
'''
    @File    :   myPool.py
    @Time    :   2022/08/11 23:56:02
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''


import torch.nn as nn
import torch as th

# 池化层 将输入的图像按照指定的方式进行池化(比如最大池化,平均池化等),操作核内范围,变成一个值
# ceil_mode:如果为True,则向上取整,会在输入图像的边界上进行池化

x = th.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]]).reshape([1,1,5,5])
x = x.type(th.float)

y = nn.functional.max_pool2d(x,kernel_size=3,stride=3,padding=0)
y2 = nn.functional.max_pool2d(x,kernel_size=3,stride=3,padding=0,ceil_mode=True)
print(y)
print(y2)