# -*- encoding: utf-8 -*-
'''
    @File    :   myConv.py
    @Time    :   2022/08/11 23:13:25
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import torch.nn as nn
import torch as th

# 卷积, nn是对nn.function的封装,只需要设置一些参数就可以了
# 卷积核与与图像上的对应位置相乘,然后相加,完成一个卷积操作,在图片上全部滑动,完成一次卷积


# conv2d 卷积函数
# input: [batch, in_channels, in_height, in_width]

# stride:步长,卷积核移动的步长
# groups:分组,卷积核的数量,一般为1,只有在分组卷积才会用到
# dialation:扩张,卷积核的扩张,一般为1,只有在图像上扩张才会用到

input = th.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]]).reshape([1,1,5,5])
kernel = th.tensor([[1,2,1],[0,1,0],[2,1,0]]).reshape([1,1,3,3])
print(input.shape,kernel.shape)


output =  nn.functional.conv2d(input,kernel,stride=1)
print(output)

output2 = nn.functional.conv2d(input,kernel,stride=2)
print(output2)

