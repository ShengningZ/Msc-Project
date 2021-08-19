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
    print("connecting mic")
##for index, name in enumerate(sr.Microphone.list_microphone_names()):
    #print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
    mic = sr.Microphone(device_index=11)
##sr.Microphone.list_microphone_names()
    print("testing3")
    with mic as source:
      r.adjust_for_ambient_noise(source,duration=1)
      audio = r.listen(source)
    print("Recognizing....")
    RecognitionResult=r.recognize_google(audio)
    print(RecognitionResult)
    print("STT finished")
    
    RecProcess=True
    break
  if RecProcess==True:
    break
