import cv2
BLACK = [0,0,0]
from helper import *
# ----------------------------------------------
# Foreground Image Processing
# ----------------------------------------------


def insert_box_data(frameData, annotation):
    mins = np.min(annotation, axis=0)
    maxs = np.max(annotation, axis=0)
    size = maxs - mins
    center = (maxs + mins) / 2
    frameData.extra_info["center"] = center.tolist()
    frameData.extra_info["size"] = size.tolist()
    return frameData



def keep_annotation(frameData, annotation):
  # return: data with images of same size but keeping only annotated part and removing outside
  frameData_copy = frameData.copy()
  cv2.fillPoly(frameData_copy.imgRGB, [annotation], BLACK)  # color in the image
  frameData_copy.imgRGB = frameData.imgRGB - frameData_copy.imgRGB

  cv2.fillPoly(frameData_copy.imgD, [annotation], BLACK)  # color in the image
  frameData_copy.imgD = frameData.imgD - frameData_copy.imgD

  frameData_copy = insert_box_data(frameData_copy, annotation)
  return frameData_copy


def get_foreground_image(frameData, annotation):
  """
  Handle image inside annotation
  """
  frameData = keep_annotation(frameData, annotation)

  # vary the brightness of the foreground RGB image
  frameData.imgRGB = frameData.imgRGB.astype(np.float32) * (np.random.random() * 0.3 + 0.85)
  frameData.imgRGB = np.clip(frameData.imgRGB, 0, 255)
  frameData.imgRGB = frameData.imgRGB.astype(np.uint8)
  return frameData

