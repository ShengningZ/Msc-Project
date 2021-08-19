import jetson.inference
import jetson.utils
from paddleocr import PaddleOCR, draw_ocr
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("/dev/video0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()

	def roi_process(img, box):

    left, bottom, right, top = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    # print(x, y, w, h)
    print("get roi!")
    #按照识别后框体的left, bottom, right, top进行切割图像，并保存为cut
    cut = img[bottom:top, left:right]
    print("cut!")

    cv2.imwrite("img", cut)
   
    w = cut.shape[1]
    cut_num = cut[:, int(3/5*w):int(4/5*w)]
    cv2.imshow("cut_num", cut_num)
    cv2.imwrite("cut_num.jpg", cut_num)
    print("wirte!")
    # cv2.imshow('cut.jpg', cut)


	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))