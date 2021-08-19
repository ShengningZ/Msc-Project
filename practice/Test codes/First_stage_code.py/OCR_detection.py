import cv2
import easyocr
import os
import time
from datetime import datetime
#cpu calculation
#reader = easyocr.Reader(['ch_sim','en'],False) #False为不使用GPU
#starttime=datetime.now()
#print('start')
#decoder 为引擎，detail 为是否显示位置信息 batch_size 设置越大，占用内存越高，识别速度越快
#result = reader.readtext(image='test.jpg',decoder='greedy',batch_size=20,detail=0) 
#endtime=datetime.now()
#print('cpu need time',(endtime-starttime).seconds,'s')

cam=cv2.VideoCapture('/dev/video0')
_,img = cam.read()
height=img.shape[0]
width=img.shape[1]

#gpu计算
reader = easyocr.Reader(['ch_sim','en'])  
starttime=datetime.now()
print('start')
result = reader.readtext(image='maxresdefault.jpg',decoder='greedy',batch_size=20,detail=0)
endtime=datetime.now()
cv2.imshow('detCam',img)
print('gpu need time',(endtime-starttime).seconds,'s')
print(result)
