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
    
