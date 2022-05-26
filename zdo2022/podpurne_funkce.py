import os
import cv2 as cv
from lxml import etree
import skimage
import skimage.color
import numpy as np

def get_annotation(filename):
    annotation={
        "filename": [], # pth.parts[-1]
        "frame_id": [],
        "object_id": [],
        "x_px": [], # x pozice obarvených hrotů v pixelech
        "y_px": [],   # y pozice obarvených hrotů v pixelech
        "annotation_timestamp": [],
    }

    tree = etree.parse(filename)

    updated = tree.xpath("//updated")[0].text # date of last change in CVAT

    for track in tree.xpath('track'):
        for point in track.xpath("points"):
            pts = point.get("points").split(",")
            x, y = pts
            annotation["filename"].append(str(filename))
            annotation["object_id"].append(track.get("id"))
            annotation["x_px"].append(x)
            annotation["y_px"].append(y)
            annotation["frame_id"].append(point.get("frame"))
            annotation["annotation_timestamp"].append(updated)
    
    return annotation

def get_image_files(dir):
    imageFiles = []

    for file in os.listdir(dir):
        if file.lower().endswith(".png"):
            imageFiles.append(file)

    return imageFiles

def scale_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv.resize(image, dim, interpolation = cv.INTER_CUBIC)
