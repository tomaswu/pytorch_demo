# -*- encoding: utf-8 -*-
'''
    @File    :   nonLiner.py
    @Time    :   2022/08/12 00:11:34
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import torch.nn as nn
import torch as th


#非线性激活
# relu  x if x>0 else 0
# sigmod x/(1+exp(-x))
# tanh  2/(1+exp(-2x))-1

x = th.tensor([1.0,-0.4,3.5]).reshape([1,-1])
y = th.nn.functional.relu(x)
print(y)

y2 = th.nn.functional.sigmoid(x)
print(y2)