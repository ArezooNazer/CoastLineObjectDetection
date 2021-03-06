#object detection on coastline using deep learning 

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2


treshHold = 0.2
pathToPrototxt = "MobileNetSSD_deploy.prototxt.txt"
pathToModel = "MobileNetSSD_deploy.caffemodel"
pathToVideo = "video/1.mp4"

# the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# load model 
net = cv2.dnn.readNetFromCaffe(pathToPrototxt,pathToModel)

# initialize the video stream, allow the cammera sensor to warmup or load video from disk
# and initialize the FPS counter
#vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(pathToVideo)
time.sleep(2.0)
fps = FPS().start()


# loop over the frames from the video stream
while True:
# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 1000 pixels
	#frame = vs.read()
	ret, frame = cap.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 300)),
		0.008, (400, 300), 128)

	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		if confidence > treshHold:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
