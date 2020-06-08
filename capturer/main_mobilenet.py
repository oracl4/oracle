import imutils
import cv2
import numpy as np
import time
from imutils.video import FPS, VideoStream
import datetime

args = {
    "prototxt":"MobileNetSSD_deploy.prototxt",
    "model": "MobileNetSSD_deploy.caffemodel",
    "confidence": 0.5
}

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

CONSIDER = set(["motorbike", "car"])
objCount = {obj: 0 for obj in CONSIDER}
frameDict = {}

print("[INFO] detecting: {}...".format(", ".join(obj for obj in CONSIDER)))

vs = VideoStream(src="rtsp://192.168.13.21:1024/h264_ulaw.sdp").start()
# vs = VideoStream(src=0).start()

totalFrames = 0

fps = FPS().start()

BoundMin = (170, 120) # Xmin, Ymin
BoundMax = (300, 270) # Xmin, Ymax

objCentroid = (0, 0)

while True:
    original_frame = vs.read()

    # Rotate Image
    (h, w) = original_frame.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    original_frame = cv2.warpAffine(original_frame, M, (w, h))

    frame = original_frame

    frame = imutils.resize(frame, width=384)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (384, 384), 127.5)

    net.setInput(blob)
    detections = net.forward()

    objCount = {obj: 0 for obj in CONSIDER}

    status = 0

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] in CONSIDER:
                objCount[CLASSES[idx]] += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                objCentroid = (int((startX+endX)/2), int((startY+endY)/2))
                y = startY - 15 if startY - 15 > 15 else startY + 15
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

                if objCentroid[0] > BoundMin[0] and objCentroid[0] < BoundMax[0] and objCentroid[1] > BoundMin[1] and objCentroid[1] < BoundMax[1]:
                    status = 1
    
    if status == 1:
        filename = "Dataset_" + datetime.datetime.now().strftime("%A%d%B%Y%I%M%S%p")
        print("Something Passing | Writing", filename)
        cv2.imwrite("dataset/" + filename + ".png", original_frame)

    cv2.circle(frame, objCentroid, 5, (0,0,255), 2)
    image = cv2.rectangle(frame, BoundMin, BoundMax, (0,255,0), 3)
    totalFrames += 1

    fps.update()

    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF
        
    if key == ord("q"):
        break

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))    

vs.stop()
cv2.destroyAllWindows()