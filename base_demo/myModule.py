# -*- encoding: utf-8 -*-
'''
    @File    :   myModule.py
    @Time    :   2022/08/11 23:04:48
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import torch.nn as nn
import torch as th

# 定义一个简单的网络
class TModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


module = TModule()
x=th.tensor(1.0)
y=module(x)
print(y)