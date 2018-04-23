import cv2
import SUNRGBD
import random as rand
import numpy as np
import os
from helper import *
from foreground import *
from background import *

BLACK = [0,0,0]
WHITE = [255,255,255]
frameDir = "/Users/jeff/Documents/cs280/project/SUNRGB/SUNRGBD/kv2/kinect2data/"
background_labels = ["FLOOR", "FLOOR1", "BATHROOMFLOOR", "STEELBARROOF", "BASEN", "BRICKDESIGNEDWALL", "WALL", "CEILING1", "STAIRS", "BATHROOMWALL", "FLLOOR", "GLASS1", "BRICKWALL", "FENSE", "BOOTGWALL"]

# ----------------------------------------------
# Loads all of the data from a given directory 
# ----------------------------------------------

#frameDir = "/Users/jeff/Documents/cs280/project/SUNRGB/SUNRGBD/kv1/NYUdata/"

def load_data(frameDir, single=False):
    """
    Looks in frameDir for all the files with a certain ending (thats how we know its an image folder) and
    reads it into a frameData object
    :param single: If True, only pulls one image out
    :return: list of frameData objects
    """
    directory = os.fsencode(frameDir)
    files = []
    for file in os.listdir(directory): # includes .DS_Store not . or ..
        filename = os.fsdecode(file)
        if filename.endswith("resize") or filename.startswith("NYU"): 
            new_file = os.path.join(directory, file)
            files.append(os.fsdecode(new_file))
    data = []                                         
    if single:
        files = [files[4]]
    for file in files:
        frameData = SUNRGBD.readFrame(file, True)
        if frameData is None:
            continue
        data.append(frameData)
    return data
                                     
def check_label_for_background(label, background_labels):
    """
    Looks at a label and determines if it is one of the many background_labels
    :param label: label string
    :param background_labels:  list of backgronud label strings
    :return: bool
    """
    label = label.upper()
    bools = [b_label in label for b_label in background_labels]
    return np.any(bools)

def save_readable(txt):
    with open(str(type(txt)), 'w') as f:
        f.write(str(txt))

data = load_data(frameDir, single=True)

# ----------------------------------------------
# Image Display 
# ----------------------------------------------

def split_image(frameData, annotation):
    """
    splits the image defined by frameData to the image just in the polygon defined by annotation and all outside of it
    :pajram frameData: full info of a scnee
    :param annotation: annotation of one object in the scene (list of points)
    :return: [image_foreground, image_background]
    """
    im_f = get_foreground_image(frameData.copy(), annotation)
    im_b = get_background_image(frameData.copy(), annotation)
    return im_f, im_b


def extract_frameData(frameData):
    """
    Given full info about one scene, find objects that are
    1. not background
    2. relatively large in size
    and split the image to two images,
    one of just that image, one of the background without it.
    """
    foreground_imgs = []
    background_imgs = []
    num_pxl_in_image = calc_size_of_image(frameData.imgRGB)
    for label, annotation in zip(frameData.labels2D, frameData.annotation2D):
        num_pxl_in_annotation = calc_size_of_annotation(frameData.imgRGB, annotation)
        annot_ratio = num_pxl_in_annotation / num_pxl_in_image 
        if annot_ratio > 0.5:  # taking up half of the scene, too big
            continue
        if annot_ratio < 0.02:  # too small 
            continue

        if check_label_for_background(label, background_labels):
            continue
        foreground_img, background_img = split_image(frameData, annotation)
        foreground_imgs.append(foreground_img)
        background_imgs.append(background_img)
    return foreground_imgs, background_imgs

def _save_image(img, name=""):
    """
    Saves a single image
    """
    cv2.imwrite(name + "RBG.png", img)

def _save_images(frameData_list, name=""):
    """
    Saves all the images in a list of frameData objects
    (Each frameData object will have RGB and D image)
    """
    for i, frameData in enumerate(frameData_list):
        imgRGB = frameData.imgRGB
        imgD = frameData.imgD
        cv2.imwrite(name + str(i) + "RBG.png", imgRGB)
        cv2.imwrite(name + str(i) + "D.png", imgD)

def save_images(data, name=""):
    """
    Save everything about all images in data, including
    1. Original RGB, D image
    2. Foreground RGB, D image
    3. Background RGB, D image
    4. Annotated RGB image
    for each object edited in each image
    """
    for i, frameData in enumerate(data):
        f, b = extract_frameData(frameData)
        _save_images(f, name + str(i) + "foreground")
        _save_images(b, name + str(i) + "background")
        _save_images([frameData], name + str(i) + "original")
        annotated_img = get_RGB_with_annotations(frameData)
        _save_image(annotated_img, name + str(i) + "annotated")

save_images(data)