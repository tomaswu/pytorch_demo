# -*- encoding: utf-8 -*-
'''
    @File    :   myDataSet.py
    @Time    :   2022/08/10 23:03:58
    @Author  :   Tomas Wu 
    @Contact :   tomaswu@qq.com
    @Desc    :   
'''

from torch.utils import data as thdata
from PIL import Image
import numpy as np
import os

class myData(thdata.Dataset):
    def __init__(self,label='ants') -> None:
        super().__init__()
        self.root_dir = './hymenoptera_data/train/'+label
        self.label_dir = label
        self.img_path=[os.path.join(self.root_dir,i) for i in os.listdir(self.root_dir)]

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img = Image.open(img_name)
        label = self.label_dir
        return np.array(img),label

    def __len__(self):
        return len(self.img_path)


#test
if __name__=='__main__':
    ants = myData()
    bees = myData('bees')
    train_data = ants + bees
    print(train_data[155])