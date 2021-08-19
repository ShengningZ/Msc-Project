import cv2
import speech_recognition as sr
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setup(13,GPIO.IN)

while True:
  X=GPIO.input(13)
  print(X)


