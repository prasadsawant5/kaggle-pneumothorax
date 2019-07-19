import cv2
import numpy as np
import os
from tqdm import tqdm
import pydicom
import tensorflow as tf
from mask_functions import *
import pandas as pd
import h5py
import sys

TRAIN = './dicom-images-train/'
TRAIN_CSV = 'train-rle.csv'
COLS = ['ImageId', 'EncodedPixels']
IMG_SIZE = (384, 384)

H5_IMAGES = './images.h5'
H5_MASKS = './masks.h5'

BATCH_SIZE = 512

if __name__ == '__main__':
    images = h5py.File(H5_IMAGES, 'r')
    truth = h5py.File(H5_MASKS, 'r')

    img_datasets = images.keys()
    truth_datasets = truth.keys()

    total_datasets = len(images.keys())

    total_loss = 0
    idx = 0

    img = images.get('images_1')
    masks = truth.get('ann_1')

    for i in range(0, BATCH_SIZE):
        batch_img = img[i]
        batch_mask = masks[i]

        cv2.imshow('Image', batch_img)
        cv2.imshow('Mask', batch_mask)

        key = cv2.waitKey()

        if key == 27:
            cv2.destroyAllWindows()
            break
