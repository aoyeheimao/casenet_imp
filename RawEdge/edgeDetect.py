"""
Performs Deep Learning based Edge Detection using HED (Holistically Nested Edge Detection)
HED uses Trimmed VGG-like CNN (for image to prediction)
Author: krshrimali
Motivation: https://cv-tricks.com/opencv-dnn/edge-detection-hed/ (by Ankit Sachan)
"""

import cv2 as cv
import numpy as np
import copy
import os
import tqdm
import glob
from preprocessing import CannyP
from preprocessing import CropLayer
from PIL import Image
import sys


class HED:
    def __init__(self) -> None:
        prototxt = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
        caffemodel = os.path.join(os.path.dirname(__file__),
                                  "hed_pretrained_bsds.caffemodel")

        cv.dnn_registerLayer('Crop', CropLayer)
        self.net = cv.dnn.readNet(prototxt, caffemodel)

    def extract_edge(self, img):
        img = np.array(img).astype(np.uint8)
        obj = CannyP(img)

        # # remove noise
        img = obj.noise_removal(filterSize=(5, 5))
        inp = cv.dnn.blobFromImage(img, scalefactor=1.0, size=(img.shape[1], img.shape[0]), \
                mean=(104.00698793, 116.66876762, 122.67891434), \
                swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()
        out = out[0, 0]
        out = cv.resize(out, (img.shape[1], img.shape[0]))
        out = 255 * out
        out = out.astype(np.uint8)
        return Image.fromarray(out)


if __name__ == "__main__":
    # # get image path
    # if (len(sys.argv) > 1):
    #     src_path = sys.argv[1]
    # else:
    #     src_path = "/Dataset/cylinder/cylinder/RealData/try4.jpg"

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img_folder = "/media/kerwin/a845ad94-53eb-4d8a-8275-8ecc1277b7cb5/cylinder/cylinder/JPEGImagesClean"
    edge_folder = "/media/kerwin/a845ad94-53eb-4d8a-8275-8ecc1277b7cb5/cylinder/cylinder/JPEGImagesEdgeClean"
    # dataset_name = "boxes"
    # source = "aaa"
    #img_folder = f"/Dataset/{dataset_name}/{dataset_name}/{source}/"
    #edge_folder = f"/Dataset/{dataset_name}/{dataset_name}/{source}Edge/"
    # img_folder = f"/Datasets/{dataset_name}/{dataset_name}/{source}/"
    # edge_folder = f"/Datasets/{dataset_name}/{dataset_name}/{source}Edge/"
    ext = "jpg"
    ext_output = "jpg"

    prototxt = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
    caffemodel = os.path.join(os.path.dirname(__file__),
                              "hed_pretrained_bsds.caffemodel")

    cv.dnn_registerLayer('Crop', CropLayer)
    net = cv.dnn.readNet(prototxt, caffemodel)

    for src_path in tqdm.tqdm(glob.glob(os.path.join(img_folder, f"*.{ext}"))):
        # read image
        img = cv.imread(src_path, 1)
        raw_img = copy.deepcopy(img)
        if (img is None):
            print("Image not read properly")
            sys.exit(0)

        # initialize preprocessing object
        obj = CannyP(img)

        # remove noise
        img = obj.noise_removal(filterSize=(5, 5))
        inp = cv.dnn.blobFromImage(img, scalefactor=1.0, size=(int(img.shape[1]/4), int(img.shape[0]/4)), \
                mean=(104.00698793, 116.66876762, 122.67891434), \
                swapRB=False, crop=False)
        net.setInput(inp)
        out = net.forward()
        out = out[0, 0]
        out = cv.erode(cv.resize(out, (img.shape[1], img.shape[0])), kernel)
        out = 255 * out
        out = out.astype(np.uint8)
        name = src_path.split("/")[-1].split(".")[0]
        cv.imwrite(os.path.join(edge_folder, f"{name}.{ext_output}"), out)
        cv.imwrite(os.path.join(img_folder, f"{name}.{ext_output}"), raw_img)
