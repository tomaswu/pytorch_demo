# -*- encoding: utf-8 -*-
'''
    @File    :   tensorboard.py
    @Time    :   2022/08/10 23:32:44
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

# 打开tensorboard
# cmd: tensorboard --logdir=logs --port=6007

import torch.utils.tensorboard as tb
from myDataSet import myData
import shutil

shutil.rmtree('./logs')

ants = myData()
mywriter = tb.SummaryWriter(log_dir='./logs')

mywriter.add_image('input',ants[10][0],1,dataformats='HWC')
mywriter.add_image('input',ants[15][0],2,dataformats='HWC')
# y=x

for i in range(100):
    mywriter.add_scalar('y=x',i,i)

mywriter.close()