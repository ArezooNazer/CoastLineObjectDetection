#first run the file coastlineObjectDetection
#then calculate the Intersection over Union (IoU) for object detection

from collections import namedtuple

# define the `Detection` object
grandTruth = namedtuple("grandTruth", ["image_path", "gt1","gt2","gt3","gt4","gt5","gt6","gt7"])
imagePath = "/grandtruth"

def bb_intersection_over_union(boxA, boxB):
	if boxA != None:
	# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
	
		# compute the area of intersection rectangle
		interArea = (xB - xA + 1) * (yB - yA + 1)
	
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
	
		# return the intersection over union value
		return iou
	else: return -1

	"""grandTruth(imagePath + "\img1.jpg", [234, 47, 616, 191]),
	grandTruth(imagePath + "\img2.jpg", [474, 161, 526, 188]),
	grandTruth(imagePath + "\img2.jpg", [1078,98, 1355, 195]),
	grandTruth(imagePath + "\img2.jpg", [928, 115, 1101,182]),
	grandTruth(imagePath + "\img3.jpg", [586, 90, 873, 203]),
	grandTruth(imagePath + "\img3.jpg", [410, 82, 601, 171]),
	grandTruth(imagePath + "\img4.jpg", [118, 148, 576, 305]),
	grandTruth(imagePath + "\img5.jpg", [373, 108, 691, 168]),"""

# define the list of example detections
examples = [
	grandTruth(imagePath + "\img16.jpg", [2461,1708,2912,1871],[2088,1674,2294,1773],[912,1654,1491,1938],[0,1477,155,2055],[303,1654,519,2012],[565,1687,869,2003],None),
	
	grandTruth(imagePath + "\img14.jpg", [3336,1608,3784,1776],[3120,1612,3334,1958],[2854,1378,3044,2010],[2238,1532,2960,1768],[2136,1620,2830,2108],None,None)]

for grandTruth in examples:
	
	image = cv2.imread(grandTruth.image_path)
	(h, w) = image.shape[:2] 
	#"""0 = h , 1 = w , 2 = c """
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (800, 800)), 0.008, (800, 800), 128)

	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()
	
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
	    objects += 1
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
		idx = int(detections[0, 0, i, 1])
		
		if confidence > treshHold :
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			
			
			# display the prediction
			"""label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("[INFO] {}".format(label))
			cv2.rectangle(image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)"""
				
			# draw the ground-truth bounding box along with the predicted
			# bounding box
			print("[INFO] computing  Intersection over Union...")
			cv2.rectangle(image, (startX, startY), 
				(endX, endY), (0, 0, 255), 5)	
			#Grand truth boxes
			cv2.rectangle(image, tuple(grandTruth.gt1[:2]), 
				tuple(grandTruth.gt1[2:]), (0, 255, 0), 5)
			if grandTruth.gt2 != None:
				cv2.rectangle(image, tuple(grandTruth.gt2[:2]), 
					tuple(grandTruth.gt2[2:]), (0, 255, 0), 5)
			if grandTruth.gt3 != None:
				cv2.rectangle(image, tuple(grandTruth.gt3[:2]), 
					tuple(grandTruth.gt3[2:]), (0, 255, 0), 5)
			if grandTruth.gt4 != None:
				cv2.rectangle(image, tuple(grandTruth.gt4[:2]), 
					tuple(grandTruth.gt4[2:]), (0, 255, 0), 5)
			if grandTruth.gt5 != None:
				cv2.rectangle(image, tuple(grandTruth.gt5[:2]), 
					tuple(grandTruth.gt5[2:]), (0, 255, 0), 5)
			if grandTruth.gt6 != None:
				cv2.rectangle(image, tuple(grandTruth.gt6[:2]), 
					tuple(grandTruth.gt6[2:]), (0, 255, 0), 5)
			if grandTruth.gt7 != None:
				cv2.rectangle(image, tuple(grandTruth.gt7[:2]), 
					tuple(grandTruth.gt4[2:]), (0, 255, 0), 5)
			
				
			
				
			# compute the intersection over union and display it
			iou1 = bb_intersection_over_union(grandTruth.gt1, (startX, startY, endX, endY))
			iou2 = bb_intersection_over_union(grandTruth.gt2, (startX, startY, endX, endY))
			iou3 = bb_intersection_over_union(grandTruth.gt3, (startX, startY, endX, endY))
			iou4 = bb_intersection_over_union(grandTruth.gt4, (startX, startY, endX, endY))
			iou5 = bb_intersection_over_union(grandTruth.gt5, (startX, startY, endX, endY))
			iou6 = bb_intersection_over_union(grandTruth.gt6, (startX, startY, endX, endY))
			iou7 = bb_intersection_over_union(grandTruth.gt7, (startX, startY, endX, endY))
			iou = max(iou1,iou2,iou3,iou4,iou5,iou6,iou7)
			finalIou = finalIou + iou
			y = startY - 10 if startY - 15 > 15 else startY + 25
			cv2.putText(image, "IoU: {:.4f}".format(iou), (endX, startY ),
				cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
			print("{}: {:.4f}".format(grandTruth.image_path, iou))
			
			# show the output image
	resized = imutils.resize(image, width=1200)
	cv2.imshow("Output", resized)
	cv2.waitKey(0)			

print("[Info] number of objects :" ,objects)
print("[Info] sum of iou :" ,finalIou)
print("[Info] mean iou is :" ,finalIou/objects)
