# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/17 13:14:20
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''
import tqdm 
import time
a=tqdm.trange(10000)


for i in range(10000):
    a.update()
    time.sleep(0.01)
