# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/02/19 11:51:28
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''
import cv2
import numpy as np

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

def calIOU(bbox1,bbox2,draw=False,show_stride=1,mesh=[13,13]):
    '''bbox format: [class id,x:TopLeft,y:TopLeft,w,h] or [x,y,w,h]'''
    # judge parameters
    assert len(bbox1)==4 or len(bbox1)==5, "wrong length of bbox1"
    assert len(bbox2)==4 or len(bbox2)==5, "wrong length of bbox2"

    # calculate iou
    x1,y1,w1,h1=bbox1[1:] if len(bbox1)==5 else bbox1
    x2,y2,w2,h2=bbox2[1:] if len(bbox2)==5 else bbox2
    if x1+w1<=x2 or x2+w2<=x1 or y1+h1<=y2 or y2+h2<=y1:
        iou=0
    else:
        x_left=min(x1+w1,x2+w2) #left x
        x_right = max(x1,x2)
        y_top=min(y1+h1,y2+h2) #left x
        y_bottom = max(y1,y2)
        w_inner = x_right-x_left
        h_inner = y_bottom-y_top
        area = w_inner*h_inner
        iou = area/(w1*h1+w2*h2-area)

    # show result
    if draw:
        x1*=show_stride
        y1*=show_stride
        w1*=show_stride
        h1*=show_stride
        x2*=show_stride
        y2*=show_stride
        w2*=show_stride
        h2*=show_stride
        b = show_stride
        img=np.zeros([b,b,3],dtype='uint8')
        assert len(mesh)==2,'mesh size error.'
        for i in range(mesh[0]+1):
            cv2.line(img,[int(i*b/mesh[0]),0],[int(i*b/mesh[0]),b],[255,255,0],1)
        for i in range(mesh[1]+1):
            cv2.line(img,[0,int(i*b/mesh[1])],[b,int(i*b/mesh[1])],[255,255,0],1)

        cv2.rectangle(img,[int(x1),int(y1)],[int(x1+w1),int(y1+h1)],[0,255,0],2)
        cv2.rectangle(img,[int(x2),int(y2)],[int(x2+w2),int(y2+h2)],[0,0,255],2)
        cv2.putText(img,f'iou:{iou:.4f}',[20,20],cv2.FONT_HERSHEY_COMPLEX,0.8,[0,255,0],1)
        cv2.imshow('test',img)
        cv2.waitKey(0)
    return iou

def anchor2bbox(anchor,stride=32,mesh = [13,13]):
    '''anchor format:[ax,ay,aw,ah]'''
    assert len(anchor)==4
    ax,ay,aw,ah = anchor
    x = (ax-aw/2)*stride*mesh[0]
    y = (ay-ah/2)*stride*mesh[1]
    w = aw*stride*mesh[0]
    h = ah*stride*mesh[1]
    return [x,y,w,h]

def padding2square(img:np.ndarray):
    h,w=img.shape[:2]
    a=max(h,w)
    c = 1 if len(img.shape)==2 else img.shape[2]
    if c==1:
        square = np.zeros([a,a],dtype='uint8')
    else:
        square = np.zeros([a,a,c],dtype='uint8')
    square[:h,:w]=img
    return square

def square2size(img:np.ndarray,bboxes,size:int=416):
    assert img.shape[0]==img.shape[1],'only support square resolution.'
    kr = size/img.shape[0]
    new_img = cv2.resize(img,[size,size])
    new_bboxes=[]
    for bbox in bboxes:
        new_bboxes.append([c if i==0 else c*kr for i,c in enumerate(bbox)])
    return new_img,new_bboxes

def xywh2owh(xywh):
    x,y,w,h=xywh
    ox=x+w/2
    oy=y+h/2
    return ox,oy,w,h

def owh2xywh(owh):
    ox,oy,w,h=owh
    x=ox-w/2
    y=oy-h/2
    return x,y,w,h 

if __name__=='__main__':
    calIOU([100,100,200,200],anchor2bbox([0.3,0.3,0.2,0.22],32),True,1,[13,13])


    