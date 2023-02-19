# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/16 20:54:53
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''
import sys, os

sys.path.append(".")

import torch as th
import torch.utils.data as tudata
from PIL import Image, ImageFile
import numpy as np
import json
import cv2
from tomas_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class COCO4YOLO(tudata.Dataset):
    def __init__(self, coco_path='/Users/tomaswu/programming/coco2017', cf='train', version=2017) -> None:
        super().__init__()
        self.__cocopath = coco_path
        self.version = version
        self.cf = cf
        self.readinfo()
        self.img_size = 416  #suqare
        self.stride = np.array([32, 16, 8]) # for 3 output size
        self.pre_case = np.array([
            [(10,13), (16,30), (33,23)],  # small obj
            [(30,61), (62,45), (59,119)],  # medium obj
            [(116,90), (156,198), (373,326)]  # large obj
        ]).reshape(-1,2)/self.img_size

    def readinfo(self):
        assert self.cf in ['train', 'val']
        jsonpath = os.path.join(self.__cocopath, f'annotations/instances_{self.cf}{self.version}.json')
        assert os.path.isfile(jsonpath), 'cant find file'
        with open(jsonpath, 'r') as f:
            self.annotations = json.load(f)
            self.images = self.annotations['images']

    def readLabel(self, image_id):
        label = []
        for i in self.annotations['annotations']:
            if i['image_id'] == image_id:
                for idx,ctx in enumerate(self.annotations['categories']):
                    if ctx['id']==i['category_id']:
                        label.append([idx] + i['bbox'])
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        fpath = os.path.join(self.__cocopath, f'{self.cf}{self.version}/{info["file_name"]}')
        base = info["file_name"].split('.')[0]
        lpath = os.path.join(self.__cocopath, f'labels/{self.cf}{self.version}/{base}.txt')
        assert os.path.isfile(fpath)
        bboxes = self.readLabel(info['id'])
        img = np.array(Image.open(fpath))
        img = img_utils.padding2square(img)
        img,bboxes = img_utils.square2size(img,bboxes,416)
        return info, img, bboxes

    def decodeLabel(self, c_id):
        for i in self.annotations['categories']:
            if i['id'] == c_id:
                return i['name']

    def rect2bbox(img, label):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        a = max(h, w)
        new_img = np.zeros_like(img, dtype='uint8')
        new_img[:h, :w] = img
        nlabel = []
        for rect in label:
            c, rx, ry, rw, rh = rect
            bx = (rx - rw / 2) / a
            by = (ry - rh / 2) / a
            bw = rw / a
            bh = rh / a
            nlabel.append([c, bx, by, bw, bh])
        return new_img, nlabel

    def bbox2label(self, bboxes):
        output_size = (self.img_size/self.stride).astype('int32')
        # label large medim large
        labels = [np.zeros([output_size[i], output_size[i], 3, 85]) for i in range(3)]
        history=[] #用来保存已经分配的的anchor
        iw = np.array([[13]*3,[26]*3,[52]*3]).flatten() #small medim large
        for i in bboxes:
            assert len(i)==5,'wrong bbox length.'
            bbox=np.array(i).reshape(-1,5)
            for j in bbox:
                b=j[1:]/self.img_size                
                cx = np.floor((b[0]+b[2]/2)*iw)
                cy = np.floor((b[1]+b[3]/2)*iw)
                tx = (b[0]+b[2]/2)*iw-cx
                ty = (b[1]+b[3]/2)*iw-cy
                pre_bs = [img_utils.owh2xywh([cx[i]/iw[i],cy[i]/iw[i],c[0],c[1]]) for i,c in enumerate(self.pre_case)]
                ious = [img_utils.calIOU(b,pb,draw=False,show_stride=416*2,mesh=[iw[i]]*2) for i,pb in enumerate(pre_bs)] #这里的draw可以绘出bbox和pre bbox的关系

                for idx,iou in enumerate(ious):
                    if iou>0.35:
                        # if history??? 暂时没有想到什么好的办法来解决bbox对应同一anchor的情况
                        onehot = [0]*80
                        onehot[int(j[0])]=1.0
                        tw=np.log(b[2]/self.pre_case[idx][0])
                        th=np.log(b[3]/self.pre_case[idx][1])
                        labels[idx//3][int(cx[idx]),int(cy[idx]),idx%3]=[tx[idx],ty[idx],tw,th,1]+onehot

                if not np.any(np.array(ious)>0.35):
                    idx = np.argmax(np.array(ious))
                    onehot = [0]*80
                    onehot[int(j[0])]=1.0
                    tw=np.log(b[2]/self.pre_case[idx][0])
                    th=np.log(b[3]/self.pre_case[idx][1])
                    labels[idx//3][int(cx[idx]),int(cy[idx]),idx%3]=[tx[idx],ty[idx],tw,th,1]+onehot

        return [np.array(i) for i in labels]





if __name__ == '__main__':
    coco = COCO4YOLO()
    for i in coco:
        info, img, label = i
        # imshow(img, bboxs=label, wait=2000, coco_dataset=coco)
        for i in coco.bbox2label(label):
            print(i.shape)
        break