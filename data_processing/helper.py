import numpy as np
import cv2
import SUNRGBD
import random as rand
# ----------------------------------------------
# Image Processing Helpers
# ----------------------------------------------
BLACK = [0,0,0]
WHITE = [255,255,255]

def find_annotated_box(im, annotation, padding=[20, 20]):
  # get the box we are working with
  pts = np.array(list(zip(*annotation)))
  top_right_initial = np.max(pts, axis=1)
  bottom_left_initial = np.min(pts, axis=1)

  top_right = [top_right_initial[0] + padding[0], top_right_initial[1] + padding[1]]
  bottom_left = [bottom_left_initial[0] - padding[0], bottom_left_initial[1] - padding[1]]

  x2, y2 = top_right
  x1, y1 = bottom_left
  x1 = max(x1, 0)
  y1 = max(y1, 0)
  return y1, y2, x1, x2 # not sure why needs to be flipped


def calc_avg_px_val(img):
  # img is a h x w x 3 image
  rgb_means = np.mean(img, axis=(0,1))
  return rgb_means


def calc_surrounding_px_val(img, annotation):
  """
  We find a loose bounding box around the annotation and find the mean pixel value outside the annotation
  """
  # Find image values around annotation
  img_helper = np.copy(img)
  img_helper = cv2.fillPoly(img_helper, [annotation], BLACK)
  x1, x2, y1, y2 = find_annotated_box(img, annotation, padding=[2,2])
  intensities = img_helper[x1:x2, y1:y2]

  w, l, c = intensities.shape
  intensities = intensities.reshape(w*l, c)
  num_pixels = w * l
  num_nonzero_pixels = np.count_nonzero(intensities, axis=0)[0]  # counts number of nonzero per channel. should be 3 of same #

  mu = np.mean(intensities, axis=0)
  # in calculating mean, we used num_pixels when we should have used num_nonzero_pixels. We make up for it now
  normalized_mu = mu * num_pixels / num_nonzero_pixels
  return normalized_mu

def calc_size_of_image(img):
  l, w, c = img.shape
  return w * l

def calc_size_of_annotation(img, annotation):
  shape = (img.shape[0], img.shape[1], 1)
  img_helper = np.zeros(shape)
  img_helper = cv2.fillPoly(img_helper, [annotation], WHITE)
  pts = np.where(img_helper == 255)
  return len(pts[0])

def replace_annotated_with_background(img, annotation):
  """
  Replace annotated part of the image with the nearest pixel from unannotated part of the image
  NOTE: THIS HAS HORRIBLE RUNTIME
  """
  shp = (img.shape[0], img.shape[1])
  img_helper = np.zeros(shp)
  img_helper = cv2.fillPoly(img_helper, [annotation], 255)
  x1, x2, y1, y2 = find_annotated_box(img, annotation, padding=[1,1])
  bitmap = img_helper == 0
  pts = np.where(bitmap[x1:x2, y1:y2])
  pts = [np.array(p) for p in zip(*pts)]

  annot_pts = np.where(img_helper == 255)

  for annot_pt in zip(*annot_pts):
    annot_pt = np.array(annot_pt)
    desired_pt = min(pts, key= lambda x: np.linalg.norm(x - annot_pt))
    img[tuple(annot_pt)] = np.copy(img[tuple(desired_pt)])

  return img



def find_all_labels(files):
  """
  This is used to help all the labels our dataset
  """
  all_labels = set()
  for file in files:
    frameData = SUNRGBD.readFrame(file, True)
    if frameData is None:
      continue
    labels2D = frameData.labels2D
    labels2D = [label.upper() for label in labels2D]
    all_labels = all_labels.union(set(labels2D))
  return all_labels


def get_RGB_with_annotations(frameData):
  """
  Given the data about our image, return the RGB image that is semantically annotated
  :param frameData:
  :return:
  """
  imgRGBWithAnnotations = np.array(frameData.imgRGB, copy=True);

  # Colors every annotation of the RGB image "uniquely"
  for i in range(0, len(frameData.annotation2D)):
    color = [rand.randint(0,255), rand.randint(0,255), rand.randint(0,255)]
    if frameData.annotation2D[i].size == 0:
      continue
    cv2.fillPoly(imgRGBWithAnnotations, [frameData.annotation2D[i]], color)

  # Labels each annotation of the RGB image
  for i in range(0, len(frameData.annotation2D)):
    data = frameData.annotation2D[i];
    if data.size == 0:
      continue
    centroid = np.mean(data,axis=0)
    cv2.putText(imgRGBWithAnnotations, frameData.labels2D[i], (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,0,0],2)
  return imgRGBWithAnnotations

def _show_image(img, name="Image"):
  cv2.imshow(name, img);


def show_image(frameData):
  cv2.imshow("Detph Image", frameData.imgD);
  cv2.imshow("RGB Image", frameData.imgRGB);
  imgRGBWithAnnotations = get_RGB_with_annotations(frameData)
  cv2.imshow("Annotated Image", imgRGBWithAnnotations);

  cv2.waitKey(0);  # I can exit out of this with a capture screen... ??
