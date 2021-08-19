import speech_recognition as sr
import RPi.GPIO as GPIO
import DefinedFunctions as df 
from threading import Thread
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
        mic = sr.Microphone(device_index=11)
    ##sr.Microphone.list_microphone_names()
        with mic as source:
            audio = r.listen(source)
        print("Recognizing....")
        df.RecognitionResult=r.recognize_google(audio)
        print(df.RecognitionResult)
        print("STT finished")
        if df.RecognitionResult=="obstacle detection":
            ODThread=Thread(target=df.ObstacleDetection(),)
            ODThread.daemon=True
            ODThread.start()
        elif df.RecognitionResult=="read for me":
            df.RunOCR()
            #TextToRead=df.TResult
            #df.ReadOut(TextToRead)
        
        RecProcess=True
        break
    

  