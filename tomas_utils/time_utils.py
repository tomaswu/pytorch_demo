# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/17 21:44:17
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import time

def TimeTest(unit='ms'):
    def g(f):
        us={'h':1/3600e9,'min':1/60e9,'s':1e-9,'ms':1e-6,'us':1e-3,'ns':1}
        assert unit in us.keys()
        def t(*args):
            t0=time.time_ns()
            r=f(*args)
            dt=time.time_ns()-t0
            print(f'{f.__name__} spent({unit}): {dt*us[unit]}')
            return r
        return t
    return g
