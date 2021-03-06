import cv2
from os import listdir
import json
import numpy as np
from copy import deepcopy as copy

class FrameData:
    def __init__(self, imgRGB,imgD,annotation2D,labels2D, extra_info={}):
        self.imgRGB = imgRGB
        self.imgD = imgD
        self.annotation2D = annotation2D
        self.labels2D = labels2D
        self.extra_info = extra_info

    def copy(self):
        new_imgRGB = copy(self.imgRGB)
        new_imgD = copy(self.imgD)
        new_annotation2D = copy(self.annotation2D)
        new_labels2D = copy(self.labels2D)
        extra_info = copy(self.extra_info)

        return FrameData(new_imgRGB, new_imgD, new_annotation2D, new_labels2D, extra_info)

    def copy_size(self):
        # copys the size over and zeros all the image
        # can use np.zeros_like
        new_imgRGB = np.zeros(self.imgRGB.shape)
        new_imgD = np.zeros(self.imgD.shape)
        new_annotation2D = copy(self.annotation2D)
        new_labels2D = copy(self.labels2D)
        extra_info = copy(self.extra_info)

        return FrameData(new_imgRGB, new_imgD, new_annotation2D, new_labels2D, extra_info)
        
def readFrame( framePath, bfx ):
    #read RGB information to numpy array    
    rgbPath = framePath + "/image/" 
    rgbPath += listdir(rgbPath)[0]
    imgRGB = cv2.imread(rgbPath);
    
    #read depth information to numpy array
    if not(bfx):
        depthPath = framePath + "/depth/" 
    else:
        depthPath = framePath + "/depth_bfx/" 
    depthPath += listdir(depthPath)[0]
    imgD = cv2.imread(depthPath);
    
    #read 2D annotations to a list o numpy arrays where each index is related with one object polygon and a list where the index links the object polygon to the object label.
    anotation2D = framePath + "/annotation2Dfinal/index.json"
    
    with open(anotation2D) as data_file:    
        try:
            data = json.load(data_file)
        except:
            return None
            
    numberOfAnot = len(data["frames"][0]["polygon"]);
    
    anootation2D = [];
    labels2D = [];
    for i in range(0,numberOfAnot):
        x = data["frames"][0]["polygon"][i]["x"]
        y = data["frames"][0]["polygon"][i]["y"]

        idxObj = data["frames"][0]["polygon"][i]["object"];
        pts2 = np.array([x,y], np.int32)
        pts2 = np.transpose(pts2);
        anootation2D.append(pts2);
        try: 
            labels2D.append(data['objects'][idxObj]["name"])
        except IndexError:
            labels2D.append("Unclassified")
        
    frameData = FrameData(imgRGB,imgD,anootation2D,labels2D)

    return frameData;
