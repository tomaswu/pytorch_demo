# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/16 08:41:32
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import torch as th
import torch.utils.data as tudata
from PIL import Image,ImageFile
import numpy as np
import os,json
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES=True

names=[]
with open('./coco_names.json','r') as f:
    dict = json.load(f)
    for i in dict.keys():
        names.append(dict[i])

def imshow(img,wait=0,name = 'test',bboxs=None):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if(bboxs):
        height,width = img.shape[:2]
        for i in bboxs:
            assert len(i)==5
            c,x,y,w,h=i
            x=int(x*width)
            y=int(y*height)
            w=int(w*width)
            h=int(h*height)
            p1=[x-w//2,y-h//2]
            p2=[x+w//2,y+h//2]
            cv2.putText(img,names[int(c)],[p1[0],p1[1]-2],cv2.FONT_HERSHEY_SIMPLEX,0.4,[0,255,0],1)
            cv2.rectangle(img,p1,p2,color=[0,225,0])

    cv2.imshow('test',img)
    cv2.waitKey(wait)

class COCO(tudata.Dataset):
    def __init__(self,coco_path = 'E:/coco2014',cf='train') -> None:
        super().__init__()
        self.__cocopath = coco_path
        self.cf=cf
        self.readinfo()
        
    def readinfo(self):
        assert self.cf in ['train','val']
        jsonpath = os.path.join(self.__cocopath,f'annotations/captions_{self.cf}2014.json')
        assert os.path.isfile(jsonpath)
        with open(jsonpath,'r') as f:
            self.annotations=json.load(f)
            self.images = self.annotations['images']

    def readLabel(self,path):
        if not os.path.isfile(path):
            return []
        with open(path,'r') as f:
            lines = f.readlines()
        for i,c in enumerate(lines):
            m=[]
            for j in c.split(' '):
                try:
                    m.append(float(j))
                except:
                    pass
            lines[i]=m
        return lines

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        fpath = os.path.join(self.__cocopath,f'train2014/{info["file_name"]}')
        base = info["file_name"].split('.')[0]
        lpath = os.path.join(self.__cocopath,f'labels/{self.cf}2014/{base}.txt')
        assert os.path.isfile(fpath)
        label  = self.readLabel(lpath)
        img=np.array(Image.open(fpath))
        return info,img,label

if __name__=='__main__':
    coco=COCO()
    print(coco.annotations.keys())