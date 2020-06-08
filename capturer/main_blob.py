# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import time
import numpy as np

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

# Capture the Video
# vs = cv2.VideoCapture("rtsp://altius:fortius@192.168.13.41/play1.sdp")
# vs = VideoStream(src="rtsp://altius:fortius@192.168.13.41/play1.sdp").start()
vs = VideoStream(src="rtsp://192.168.13.21:1024/h264_ulaw.sdp").start()

# initialize the first frame in the video stream
firstFrame = None

last_time = int(round(time.time() * 1000))

print("Start Capturing!")

# loop over the frames of the video
while True:
	time_now = int(round(time.time() * 1000))
	# grab the current frame and initialize the occupied/unoccupied
	# text
	original_frame = vs.read()
	
	# get image height, width
	(h, w) = original_frame.shape[:2]
	
	# calculate the center of the image
	center = (w / 2, h / 2)

	M = cv2.getRotationMatrix2D(center, 180, 1.0)
	original_frame = cv2.warpAffine(original_frame, M, (w, h))
	
	frame = original_frame

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue
	
	# if ((time_now - last_time) > 60000):
	# 	firstFrame = gray
	
	status = 0
	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 6000:
			continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		status = 1

		
	if status == 1 and ((time_now - last_time) > 1000):
		filename = "Dataset_" + datetime.datetime.now().strftime("%A%d%B%Y%I%M%S%p")
		print("Something Passing | Writing", filename)
		last_time = time_now
		# original_frame = white_balance(original_frame)
		cv2.imwrite("dataset/" + filename + ".png", original_frame)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
vs.stream.release()
cv2.destroyAllWindows()
