# import cv2
# import SUNRGBD
# import random as rand
# import numpy as np
import os
import os.path as osp
from os.path import join
import json
# from helper import *
from foreground import *
from background import *
import shutil, errno
from random import shuffle


# import os.path as osp

BLACK = [0,0,0]
WHITE = [255,255,255]
frameDir = "/home/ryan/cs/datasets/SUNRGBD/kv2/kinect2data/"
background_labels = ["FLOOR", "FLOOR1", "BATHROOMFLOOR", "STEELBARROOF", "BASEN", "BRICKDESIGNEDWALL", "WALL", "CEILING1", "STAIRS", "BATHROOMWALL", "FLLOOR", "GLASS1", "BRICKWALL", "FENSE", "BOOTGWALL"]

test_split =0.95
# ----------------------------------------------
# Loads all of the data from a given directory 
# ----------------------------------------------

#frameDir = "/Users/jeff/Documents/cs280/project/SUNRGB/SUNRGBD/kv1/NYUdata/"
#frameDir = "/home/ubuntu/SUNRGBD/kv2/kinect2data/"

# https://stackoverflow.com/questions/1994488/copy-file-or-directories-recursively-in-python
def copy_dir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def load_data(frameDir, num_samples=None, start_index=49):
    """
    Looks in frameDir for all the files with a certain ending (thats how we know its an image folder) and
    reads it into a frameData object
    :param single: If True, only pulls one image out
    :return: list of frameData objects
    """
    #directory = os.fsencode(frameDir)
    directory = frameDir
    files = []
    for file in os.listdir(directory): # includes .DS_Store not . or ..
        #filename = os.fsdecode(file)
        filename = file
        if filename.endswith("resize") or filename.startswith("NYU"): 
            new_file = os.path.join(directory, file)
            #files.append(os.fsdecode(new_file))
            files.append(new_file)
    data = []                                         
    if num_samples is not None:
        files = files[start_index:start_index+num_samples]
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
        if annotation.size == 0:
            continue
        num_pxl_in_annotation = calc_size_of_annotation(frameData.imgRGB, annotation)
        annot_ratio = 1.0 * num_pxl_in_annotation / num_pxl_in_image 
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
        # Merge the RGB and depth image together
        imgD = imgD[:,:,0]
        full_img = np.dstack((imgRGB, imgD))
        cv2.imwrite(name + "_" + str(i) + ".png", full_img)

        if len(frameData.extra_info) != 0:
            with open(name + "_" + str(i) + "_bounding_box.json", "w") as fp:
                json.dump(frameData.extra_info, fp)

def save_images(data, dataset_name=""):
    """
    Save everything about all images in data, including
    1. Original RGB, D image
    2. Foreground RGB, D image
    3. Background RGB, D image
    4. Annotated RGB image
    for each object edited in each image
    """
    split_index = int(test_split * len(data))

    for d in ['train', 'validation']:
        for f in ['foreground', 'background', 'original', 'annotated']:
            if not os.path.isdir(join(d, f)):
                os.makedirs(join(d, f))


    for i, frameData in enumerate(data):
        # datum_name = name + str(i)

        if i < split_index:
            tt_dir = 'train/'
        else:
            tt_dir = 'validation/'

        try: 
            f, b = extract_frameData(frameData)
            name = dataset_name + str(i)
            foreground_name = osp.join(tt_dir + "foreground", name)
            background_name = osp.join(tt_dir + "background", name)
            original_name = osp.join(tt_dir + "original", name)
            annotated_name = osp.join(tt_dir + "annotated", name)
            _save_images(f, foreground_name)
            _save_images(b, background_name)
            _save_images([frameData], original_name)
            annotated_img = get_RGB_with_annotations(frameData)
            _save_image(annotated_img, annotated_name)
            print("Finished processing image " + str(i))
        except Exception as e:
            print("Unable to process image " + str(i))
            print(e)

    copy_dir('validation', 'test')
    os.rename('test/original', 'test/background')

    original_names = os.listdir('test/background')
    shuffle(original_names)
    for i, img in enumerate(os.listdir('test/background')):
        shutil.move(join('test/background', img), join('test/background', original_names[i]))

if __name__ == "__main__":
    data = load_data(frameDir, num_samples=None)
    save_images(data)
