# -*- encoding: utf-8 -*-
'''
    @File    :   gpu_test.py
    @Time    :   2022/08/13 22:23:51
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import torch as th

print(th.backends.mps.is_available())
print(th.backends.mps.is_built())