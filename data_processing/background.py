import numpy as np
import cv2
from helper import *
# ----------------------------------------------
# Background Image Processing
# ----------------------------------------------


def blur_annotated_area(im, annotation):
  l, w, _ = im.shape

  x1, x2, y1, y2 = find_annotated_box(im, annotation)

  # Pick out the subimage we want to work with
  subim = im [x1:x2, y1:y2]

  # Apply strong Gaussian blurring onto the subimage
  subim_blurred = cv2.GaussianBlur(subim, (5,5), 20)
  subim_blurred = cv2.GaussianBlur(subim_blurred, (5,5), 20)
  subim_blurred = cv2.GaussianBlur(subim_blurred, (7,7), 20)
  subim_blurred = cv2.GaussianBlur(subim_blurred, (7,7), 20)
  subim_blurred = cv2.GaussianBlur(subim_blurred, (7,7), 20)
  subim_blurred = cv2.GaussianBlur(subim_blurred, (3,3), 2)

  # Put it back into the original image
  im[x1:x2, y1:y2] = subim_blurred

  # Blur the image slightly
  #im = cv2.GaussianBlur(im, (7, 7), 10)
  return im

def get_background_image(frameData, annotation):
  # Take the colors surrounding the annotated box
  color_mean = calc_surrounding_px_val(frameData.imgRGB, annotation)
  cv2.fillPoly(frameData.imgRGB, [annotation], color_mean)  # color out object in background image

  # Take the depth surrounding the annotated box
  depth_background = calc_surrounding_px_val(frameData.imgD, annotation)
  cv2.fillPoly(frameData.imgD, [annotation], depth_background)  # color out the object in depth image

  # blur the background rgb image
  frameData.imgRGB = blur_annotated_area(frameData.imgRGB, annotation)
  return frameData
