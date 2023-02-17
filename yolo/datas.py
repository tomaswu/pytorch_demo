# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/16 20:54:53
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''
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

def imshow(img,wait=0,name = 'test',bboxs=None,coco_dataset=None):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if bboxs:
        height,width = img.shape[:2]
        for i in bboxs:
            assert len(i)==5
            c,x,y,w,h=i
            p1=[int(x),int(y)]
            p2=[int(x+w),int(y+h)]
            cs=coco_dataset.decodeLabel(c) if coco_dataset else str(c)
            cv2.putText(img,cs,[p1[0],p1[1]-2],cv2.FONT_HERSHEY_SIMPLEX,0.4,[0,255,0],1)
            cv2.rectangle(img,p1,p2,color=[0,225,0])

    cv2.imshow('test',img)
    cv2.waitKey(wait)

class COCO(tudata.Dataset):
    def __init__(self,coco_path = '/Users/tomaswu/programming/coco2017',cf='train',version=2017) -> None:
        super().__init__()
        self.__cocopath = coco_path
        self.version=version
        self.cf=cf
        self.readinfo()
        
    def readinfo(self):
        assert self.cf in ['train','val']
        jsonpath = os.path.join(self.__cocopath,f'annotations/instances_{self.cf}{self.version}.json')
        assert os.path.isfile(jsonpath),'cant find file'
        with open(jsonpath,'r') as f:
            self.annotations=json.load(f)
            self.images = self.annotations['images']

    def readLabel(self,image_id):
        label=[]
        for i in self.annotations['annotations']:
            if i['image_id']==image_id:
                label.append([i['category_id']]+i['bbox'])
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        fpath = os.path.join(self.__cocopath,f'{self.cf}{self.version}/{info["file_name"]}')
        base = info["file_name"].split('.')[0]
        lpath = os.path.join(self.__cocopath,f'labels/{self.cf}{self.version}/{base}.txt')
        assert os.path.isfile(fpath)
        label  = self.readLabel(info['id'])
        img=np.array(Image.open(fpath))
        return info,img,label
    
    def decodeLabel(self,c_id):
        for i in self.annotations['categories']:
            if i['id']==c_id:
                return i['name']
    
if __name__=='__main__':      
    coco=COCO()
    for i in coco:
        info,img,label=i
        imshow(img,bboxs=label,wait=2000,coco_dataset=coco)