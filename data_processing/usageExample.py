import cv2
import SUNRGBD
import random as rand
import numpy as np
import os

# ----------------------------------------------
# Loads all of the data from a given directory 

frameDir = "/Users/jeff/Documents/cs280/project/SUNRGB/SUNRGBD/kv2/kinect2data/"
#frameDir = "/Users/jeff/Documents/cs280/project/SUNRGB/SUNRGBD/kv1/NYUdata/"
directory = os.fsencode(frameDir)

files = []
for file in os.listdir(directory): # includes .DS_Store not . or ..
    filename = os.fsdecode(file)
    if filename.endswith("resize") or filename.startswith("NYU"): 
        new_file = os.path.join(directory, file)
        files.append(os.fsdecode(new_file))

frameData = SUNRGBD.readFrame(files[1234], True)
        

# ----------------------------------------------
# Image Processing 

imgRGB = frameData.imgRGB
imgD = frameData.imgD
labels2D = frameData.labels2D
annotation2D = frameData.annotation2D

imgRGBWithAnnotations = np.array(frameData.imgRGB, copy=True);

# Colors every annotation of the RGB image "uniquely"
for i in range(0, len(frameData.annotation2D)):
	color = [rand.randint(0,255), rand.randint(0,255), rand.randint(0,255)]
	cv2.fillPoly(imgRGBWithAnnotations, [frameData.annotation2D[i]], color)

# Labels each annotation of the RGB image 
for i in range(0, len(frameData.annotation2D)):	
	data = frameData.annotation2D[i];
	centroid = np.mean(data,axis=0)
	cv2.putText(imgRGBWithAnnotations, frameData.labels2D[i], (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,0,0],2)
	
cv2.imshow("Detph Image", frameData.imgD);
cv2.imshow("RGB Image", frameData.imgRGB);
cv2.imshow("Annotated Image", imgRGBWithAnnotations);

cv2.waitKey(0);  # I can exit out of this with a capture screen... ??
