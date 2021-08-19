import jetson.inference
import jetson.utils

import argparse
import sys
import os

import cv2
import re
import numpy as np

import io
import time
import json
import random

import pyrealsense2 as rs

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
 jetson.utils.logUsage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
		help="pre-trained model to load (see below for options)")

parser.add_argument("--threshold", type=float, default=0.5,
		help="minimum detection threshold to use")

parser.add_argument("--width", type=int, default=640,
		help="set width for image")

parser.add_argument("--height", type=int, default=480,
		help="set height for image")

opt = parser.parse_known_args()[0]

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, opt.width, opt.height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, opt.width, opt.height, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)

press_key = 0
while (press_key==0):
	
# Wait for a coherent pair of frames: depth and color

	frames = pipeline.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()
	if not depth_frame or not color_frame:
		continue

	
    # Convert images to numpy arrays

	depth_image = np.asanyarray(depth_frame.get_data())
	show_img = np.asanyarray(color_frame.get_data())
    # convert to CUDA (cv2 images are numpy arrays, in BGR format)
	bgr_img = jetson.utils.cudaFromNumpy(show_img, isBGR=True)

    # convert from BGR -> RGB
	img = jetson.utils.cudaAllocMapped(width=bgr_img.width,height=bgr_img.height,format='rgb8')
	jetson.utils.cudaConvertColor(bgr_img, img)
    # detect objects in the image (with overlay)
	detections = net.Detect(img)

	for num in range(len(detections)) :
		score = round(detections[num].Confidence,2)
		box_top=int(detections[num].Top)
		box_left=int(detections[num].Left)
		box_bottom=int(detections[num].Bottom)
		box_right=int(detections[num].Right)
		box_center=detections[num].Center
		label_name = net.GetClassDesc(detections[num].ClassID)

		point_distance=0.0
		for i in range (10):
			point_distance = point_distance + depth_frame.get_distance(int(box_center[0]),int(box_center[1]))

		point_distance = np.round(point_distance / 10, 3)

		distance_text = str(point_distance) + 'm'

		cv2.rectangle(show_img,(box_left,box_top),(box_right,box_bottom),(125,125,125),2)

		cv2.line(show_img,
			(int(box_center[0])-10, int(box_center[1])),
			(int(box_center[0]+10), int(box_center[1])),
			(0, 255, 255), 3)
		cv2.line(show_img,
			(int(box_center[0]), int(box_center[1]-10)),
			(int(box_center[0]), int(box_center[1]+10)),
			(0, 255, 255), 3)

		cv2.putText(show_img,
			label_name + ' ' + distance_text,
			(box_left+5,box_top+20),cv2.FONT_HERSHEY_SIMPLEX,1,
			(0,0,255),2,cv2.LINE_AA)
	cv2.putText(show_img,
		"{:.0f} FPS".format(net.GetNetworkFPS()),
		(int(opt.width*0.8), int(opt.height*0.1)),
		cv2.FONT_HERSHEY_SIMPLEX,1,
		(0,255,255),2,cv2.LINE_AA)
	display = cv2.resize(show_img,(int(opt.width*1.5),int(opt.height*1.5)))

	cv2.imshow('Detecting...',display)

	keyValue=cv2.waitKey(1)
	if keyValue & 0xFF == ord('q'):
		press_key=1
cv2.destroyAllWindows()
pipeline.stop()
