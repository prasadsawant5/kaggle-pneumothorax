import cv2
import numpy as np
import os
from tqdm import tqdm
import pydicom
from mask_functions import *
import pandas as pd

TRAIN = './dicom-images-train/'

TEST_FILE = '1.2.276.0.7230010.3.1.4.8323329.12743.1517875241.599591.dcm'
TEST_CSV = 'train-rle-sample.csv'

if __name__ == '__main__':
    dataset = pydicom.dcmread(TEST_FILE)

    # print(dataset)

    img = dataset.pixel_array * 255

    df = pd.read_csv(TEST_CSV, sep=',')
    df.columns = ['a', 'b']
    msk = df.iloc[8]['b']

    mask = rle2mask(msk, img.shape[1], img.shape[0])

    cv2.imshow('sample image dicom', img)
    cv2.imshow('Mask', mask)

    cv2.waitKey()

    # for d in tqdm(os.listdir(TRAIN)):
    #     if ',' in d:
    #         print('Folder: {}'.format(d))

    #     for img in os.listdir(TRAIN + d):
    #         if ',' in img:
    #             print('File: {}'.format(TRAIN + d + img))