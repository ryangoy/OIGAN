import cv2
BLACK = [0,0,0]
from helper import *

# ----------------------------------------------
# Foreground Image Processing
# ----------------------------------------------
def keep_annotation(frameData, annotation):
  # return: data with images of same size but keeping only annotated part and removing outside
  frameData_copy = frameData.copy()
  cv2.fillPoly(frameData_copy.imgRGB, [annotation], BLACK)  # color in the image
  frameData_copy.imgRGB = frameData.imgRGB - frameData_copy.imgRGB

  cv2.fillPoly(frameData_copy.imgD, [annotation], BLACK)  # color in the image
  frameData_copy.imgD = frameData.imgD - frameData_copy.imgD

  return frameData_copy


def get_foreground_image(frameData, annotation):
  """
  Handle image inside annotation
  """
  frameData = keep_annotation(frameData, annotation)

  # vary the brightness of the foreground RGB image
  frameData.imgRGB = frameData.imgRGB.astype(np.float32) * (np.random.random() * 0.3 + 0.85)
  frameData.imgRGB = frameData.imgRGB.astype(np.uint8)
  return frameData

