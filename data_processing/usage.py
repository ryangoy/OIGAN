import cv2
import SUNRGBD
import random as rand
import numpy as np
import os

BLACK = [0,0,0]
WHITE = [255,255,255]
frameDir = "/Users/jeff/Documents/cs280/project/SUNRGB/SUNRGBD/kv2/kinect2data/"
background_labels = ["FLOOR", "FLOOR1", "BATHROOMFLOOR", "STEELBARROOF", "BASEN", "BRICKDESIGNEDWALL", "WALL", "CEILING1", "STAIRS", "BATHROOMWALL", "FLLOOR", "GLASS1", "BRICKWALL", "FENSE", "BOOTGWALL"]

# ----------------------------------------------
# Loads all of the data from a given directory 
# ----------------------------------------------

#frameDir = "/Users/jeff/Documents/cs280/project/SUNRGB/SUNRGBD/kv1/NYUdata/"

def load_data(frameDir, single=False):
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
                                     
def find_all_labels(files):
    all_labels = set()
    for file in files:
        frameData = SUNRGBD.readFrame(file, True)
        if frameData is None:
            continue
        labels2D = frameData.labels2D
        labels2D = [label.upper() for label in labels2D]
        all_labels = all_labels.union(set(labels2D))
    return all_labels

def check_label_for_background(label, background_labels):
    label = label.upper()
    bools = [b_label in label for b_label in background_labels]
    return np.any(bools)

def save_readable(txt):
    with open(str(type(txt)), 'w') as f:
        f.write(str(txt))

data = load_data(frameDir, single=True)

# ----------------------------------------------
# Image Processing 
# ----------------------------------------------

def get_RGB_with_annotations(frameData):
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
    return imgRGBWithAnnotations
	
def _show_image(img, name="Image"):
    cv2.imshow(name, img);


def show_image(frameData):
    cv2.imshow("Detph Image", frameData.imgD);
    cv2.imshow("RGB Image", frameData.imgRGB);
    imgRGBWithAnnotations = get_RGB_with_annotations(frameData)
    cv2.imshow("Annotated Image", imgRGBWithAnnotations);

    cv2.waitKey(0);  # I can exit out of this with a capture screen... ??


def keep_annotation(frameData, annotation):
    # return: data with images of same size but keeping only annotated part and removing outside
    frameData_copy = frameData.copy()
    cv2.fillPoly(frameData_copy.imgRGB, [annotation], BLACK)  # color in the image 
    frameData_copy.imgRGB = frameData.imgRGB - frameData_copy.imgRGB

    cv2.fillPoly(frameData_copy.imgD, [annotation], BLACK)  # color in the image 
    frameData_copy.imgD = frameData.imgD - frameData_copy.imgD

    return frameData_copy


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


def split_image(frameData, annotation):
    # splits the image defined by frameData to the image just in the polygon defined by annotation and all outside of it
    # return: [image_in_annotation, image_outside_annotation]
    # Handle image inside annotation
    im_i = keep_annotation(frameData, annotation)

    # vary the brightness of the foreground RGB image
    im_i.imgRGB = im_i.imgRGB.astype(np.float32) * np.random.random() * (0.3) + 0.85
    im_i.imgRGB = im_i.imgRGB.astype(np.uint8)

    # Handle image outside annotation
    im_o = frameData.copy()


#frameData.imgRGB = replace_annotated_with_background(frameData.imgRGB, annotation)
    color_mean = calc_surrounding_px_val(frameData.imgRGB, annotation)
    cv2.fillPoly(im_o.imgRGB, [annotation], color_mean)  # color out object in background image

    depth_background = calc_surrounding_px_val(frameData.imgD, annotation)
    cv2.fillPoly(im_o.imgD, [annotation], depth_background)  # color out the object in depth image

    # blur the background rgb image 
    im_o.imgRGB = blur_annotated_area(im_o.imgRGB, annotation)
    
    return im_i, im_o

def calc_avg_px_val(img):
    # img is a h x w x 3 image
    rgb_means = np.mean(img, axis=(0,1)) 
    return rgb_means


def calc_surrounding_px_val(img, annotation):
    # TODO I mean this is OK, but would be nice if we had a nearest pixel techquie
    # Find image values around annotation
    img_helper = np.copy(img)
    img_helper = cv2.fillPoly(img_helper, [annotation], BLACK) 
    x1, x2, y1, y2 = find_annotated_box(img, annotation, padding=[2,2])
    intensities = img_helper[x1:x2, y1:y2]

    px = np.mean(intensities, axis=(0,1))
    return px 

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


def extract_frameData(frameData):
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
    cv2.imwrite(name + "RBG.png", img)

def _save_images(frameData_list, name=""):
    for i, frameData in enumerate(frameData_list):
        imgRGB = frameData.imgRGB
        imgD = frameData.imgD
        cv2.imwrite(name + str(i) + "RBG.png", imgRGB)
        cv2.imwrite(name + str(i) + "D.png", imgD)

def save_images(data, name=""):
    for i, frameData in enumerate(data):
        f, b = extract_frameData(frameData)
        _save_images(f, name + str(i) + "foreground")
        _save_images(b, name + str(i) + "background")
        _save_images([frameData], name + str(i) + "original")
        annotated_img = get_RGB_with_annotations(frameData)
        _save_image(annotated_img, name + str(i) + "annotated")

save_images(data)
