import torchvision
from torchvision.models import detection
import numpy as np
import argparse
import torch
import cv2
import json

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to the input image")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = ['person']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the model and set it to evaluation mode
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

# load the image from disk
image = cv2.imread(args["image"])
orig = image.copy()
# convert the image from BGR to RGB channel ordering and change the
# image from channels last to channels first ordering
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))
# add the batch dimension, scale the raw pixel intensities to the
# range [0, 1], and convert the image to a floating point tensor
image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)
# send the input to the device and pass the it through the network to
# get the detections and predictions
image = image.to(DEVICE)
detections = model(image)[0]
a = []


# loop over the detections
for i in range(0, len(detections["boxes"])):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections["scores"][i]
	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
	if confidence > args["confidence"] and int(detections["labels"][i]) == 1:
		# extract the index of the class label from the detections,
		# then compute the (x, y)-coordinates of the bounding box
		# for the object
		idx = int(detections["labels"][i])
		box = detections["boxes"][i].detach().cpu().numpy()
		(startX, startY, endX, endY) = box.astype("int")
		a.append({ 'image': args["image"], 'bbox': [int(startX), int(startY), int(endX), int(endY)]})
		print(a)
		# # display the prediction to our terminal
		# label = "{}: {:.2f}%".format(CLASSES[0], confidence * 100)
		# print("[INFO] {}".format(label))
		# # draw the bounding box and label on the image
		# cv2.rectangle(orig, (startX, startY), (endX, endY),
		# 	0, 2)
		# y = startY - 15 if startY - 15 > 15 else startY + 15
		# cv2.putText(orig, label, (startX, y),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
		

with open('person_boxes.json','r+') as file:
	# First we load existing data into a dict.
	file_data = json.load(file)
	# Join new_data with file_data inside emp_details
	file_data["boxes"].extend(a)
	# Sets file's current position at offset.
	file.seek(0)
	# convert back to json.
	json.dump(file_data, file, indent = 4)
# show the output image
# print(len(detections['boxes']))
# cv2.imshow("Output", orig)
# cv2.waitKey(0)