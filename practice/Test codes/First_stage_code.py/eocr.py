import cv2
import easyocr
import os
import time
from datetime import datetime

print(cv2.__version__)
dispW=640
dispH=360
flip=2

#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam= cv2.VideoCapture(camSet)

cam=cv2.VideoCapture(0)
while True:
    ret, img = cam.read()
    cv2.imshow('nanoCam',img)
    # 展示图片
    cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转灰度图
    cv2.imshow('capture',img)
    # 保存图片
    cv2.imwrite(r'readpic' + '.png',img)
    reader = easyocr.Reader(['ch_sim','en'])  
    starttime=datetime.now()
    print('start')
    result = reader.readtext(image='redpic.png',decoder='greedy',batch_size=20,detail=0)
    endtime=datetime.now()
    cv2.imshow('detCam',img)
    print('gpu needs time',(endtime-starttime).seconds,'s')
    print(result)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()