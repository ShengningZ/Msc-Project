import os
import pyttsx3
from gtts import gTTS

text='Now it is working'
myOutput=gTTS(text=text,lang='en',slow=False)
myOutput.save('talk.mp3')
os.system('mpg123 talk.mp3')

