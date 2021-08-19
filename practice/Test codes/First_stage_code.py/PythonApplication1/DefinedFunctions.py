from time import *
from threading import Thread
import cv2
#input the text and read for u
def ReadOut(TextToRead:str):
    import os
    import pyttsx3
    from gtts import gTTS
    
    text=TextToRead
    myOutput=gTTS(text=text,lang='en',slow=False)
    myOutput.save('talk.mp3')
    os.system('mpg123 talk.mp3')



#press button to start recognizing and output a recognition result
def PressAndRecognize():
    import speech_recognition as sr
    import RPi.GPIO as GPIO

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(15,GPIO.IN)
    RecProcess=0

    while True:
        X=GPIO.input(15)
        print(X)
  
        while X==0:
            Press=True
            #start speech
            print("start speaking")
            r = sr.Recognizer()
        ##for index, name in enumerate(sr.Microphone.list_microphone_names()):
        ##print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
            mic = sr.Microphone()
        ##sr.Microphone.list_microphone_names()
            with mic as source:
                audio = r.listen(source)
            print("Recognizing....")
            RecognitionResult=r.recognize_google(audio)
            print(RecognitionResult)
            print("STT finished")
            RecProcess=True
            break
        if cv2.waitKey(1)==ord('q'):
            break
    return RecognitionResult


#returns the item and its location
def ObstacleDetection():
    import jetson.inference
    import jetson.utils
    import time
    import cv2
    import numpy as np 

    timeStamp=time.time()
    fpsFilt=0
    net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.5)
    dispW=640
    dispH=360
    flip=2
    font=cv2.FONT_HERSHEY_SIMPLEX

    cam=cv2.VideoCapture('/dev/video0')
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
    print('preparing to detect')

    while True:
        #img, width, height= cam.CaptureRGBA()
        _,img = cam.read()
        height=img.shape[0]
        width=img.shape[1]

        frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
        frame=jetson.utils.cudaFromNumpy(frame)
        odoutput="Start"
        detections=net.Detect(frame, width, height)
        for detect in detections:
            #print(detect)
            RecognitionResult=" "
            ID=detect.ClassID
            top=detect.Top
            left=detect.Left
            bottom=detect.Bottom
            right=detect.Right
            item=net.GetClassDesc(ID)
            center=(left+right/2)
            print(item,center)
            
            if center<=213:
                odoutput=item+" is in the left front of you"
                print(item," is in the left front of you")
            elif center>213 and center<=426:
                odoutput=item+" is in the middle front of you"
                print(item," is in the middle front of you")
            elif center>426:
                odoutput=item+" is in the right front of you"
                print(item," is in the right front of you")
        #display.RenderOnce(img,width,height)
        dt=time.time()-timeStamp
        timeStamp=time.time()
        fps=1/dt
        fpsFilt=.9*fpsFilt + .1*fps
        #print(str(round(fps,1))+' fps')
        cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(0,0,255),2)
        cv2.imshow('detCam',img)
        cv2.moveWindow('detCam',0,0)
        #R=Thread(target=ReadOut,args=(TextToRead,))
        #R.daemon=True
        #R.start()
        if cv2.waitKey(1)==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    return odoutput



#grab a pic and out put words in it
def RunOCR():
    import cv2
    from aip import AipOcr

    """ Your APPID AK SK """
    APP_ID = '24601557'
    API_KEY = 'ykK8brcWggHf6OLuG56vOoeq'
    SECRET_KEY = 'LqxksXfNQ5nCMLB2fd1zuGc2tKsorG9T'



    client=AipOcr(APP_ID,API_KEY,SECRET_KEY)
    def get_file_content(file_path):
        #readpic
        with open(file_path, 'rb') as fp:
            return fp.read()
        global TResult
    #set up camera
    cam=cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        # save pic
        cv2.imwrite(r'readpic' + '.png',img)
        break
    
    #grabpic and run
    print('grabbing the pic')
    image = get_file_content('readpic.png')
    res=client.general(image)

    print('reading')
    for item in res['words_result']:
        print(item['words'])
        TResult=item['words']
    return TResult
    
#RunOCR()

#PressAndRecognize()

ObstacleDetection()
