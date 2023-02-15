# -*- encoding: utf-8 -*-
'''
    @File    :   myDataLoader.py
    @Time    :   2022/08/11 22:37:37
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

import multiprocessing
import torch.utils.data as td
import torch.utils.tensorboard as tboard
import torchvision as tv
import PIL
import shutil
try:
    shutil.rmtree('./logs')
except:
    pass

#data loader
# batch_size:每次返回的数据量
# shuffle:是否打乱数据
# num_workers:加载数据的线程数
# batch_sampler:每次返回的数据量
# collate_fn:每次返回的数据量
# drop_last:是否丢弃最后一个不够batch_size的数据
# pin_memory:是否将数据加载到CUDA上

test_data = tv.datasets.CIFAR10('./CIFAR10/',train=False,download=True,transform=tv.transforms.ToTensor())

test_loader = td.DataLoader(test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

img,target = test_data[0]
print(img.shape)
print(target)

if __name__=='__main__':
    writer = tboard.SummaryWriter('./logs')
    multiprocessing.freeze_support()
    step=0
    for data in test_loader:
        img,target=data
        writer.add_images('test_data',img,step)
        print(img.shape)
        print(target)
        step+=1
    print('finished!')
    writer.close()
   


