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

TF_RECORD_IMAGES = './images.tfrecord'
TF_RECORD_MASKS = './masks.tfrecord'

H5_IMAGES = './images.h5'
H5_MASKS = './masks.h5'

LABELS = './labels-train'

BATCH_SIZE = 512

use_h5 = True

def extract_images(data_record):
    features = {
        'image': tf.FixedLenFeature([], tf.string)
    }

    sample = tf.parse_single_example(data_record, features)

    img = sample['image']

    return img

def extract_annotations(data_record):
    features = {
        'annotation': tf.FixedLenFeature([], tf.string)
    }

    sample = tf.parse_single_example(data_record, features)

    ann = sample['annotation']

    return ann

def bytes_feature(value):
    """
        Returns a bytes_list from a string / byte.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_image(img, isAnn=False):
    example = None

    if isAnn:
        example = tf.train.Example(features = tf.train.Features(feature = {
            'annotation': bytes_feature(img)
        }))
    else:
        example = tf.train.Example(features = tf.train.Features(feature = {
            'image': bytes_feature(img)
        }))

    return example

if __name__ == '__main__':
    df = pd.read_csv(TRAIN_CSV)

    if use_h5:
        if not os.path.isfile(H5_IMAGES) and not os.path.isfile(H5_MASKS):
            images = h5py.File(H5_IMAGES, 'w')
            truth = h5py.File(H5_MASKS, 'w')

            img_raw = []
            truth_raw = []

            idx = 1
            f_id = 0

            for d in tqdm(os.listdir(TRAIN)):
                for sub_dir in os.listdir(TRAIN + d):
                    for f in os.listdir(TRAIN + d + '/' + sub_dir):
                        f_name = f.split('.dcm')[0]

                        img = pydicom.dcmread(TRAIN + d + '/' + sub_dir + '/' + f)
                        img = img.pixel_array

                        rle = df[df[COLS[0]] == f_name].values[0][1]                
                        
                        mask = []
                        if len(rle) > 3:
                            mask = rle2mask(rle, img.shape[1], img.shape[0])
                        else:
                            mask = np.zeros((img.shape[0], img.shape[1]))

                        img = cv2.resize(img, IMG_SIZE) * (1.0 / 255.0)
                        mask = cv2.resize(mask, IMG_SIZE) * (1.0 / 255.0)

                        img_raw.append(img)
                        truth_raw.append(mask)

                        if (idx == BATCH_SIZE) or (idx == 435 and f_id == 20):
                            img_raw = np.array(img_raw)
                            img_raw = np.reshape(img_raw, (img_raw.shape[0], img_raw.shape[1], img_raw.shape[2], 1))
                            
                            truth_raw = np.array(truth_raw)
                            truth_raw = np.reshape(truth_raw, (truth_raw.shape[0], truth_raw.shape[1], truth_raw.shape[2], 1))

                            images.create_dataset('images_{}'.format(f_id), (len(img_raw), IMG_SIZE[0], IMG_SIZE[1], 1), np.float32)
                            images['images_{}'.format(f_id)][...] = img_raw
                            img_raw = []

                            truth.create_dataset('ann_{}'.format(f_id), (len(truth_raw), IMG_SIZE[0], IMG_SIZE[1], 1), np.float32)
                            truth['ann_{}'.format(f_id)][...] = truth_raw
                            truth_raw = []

                            f_id += 1
                            idx = 0

                        idx += 1
    else:
        if not os.path.isfile(TF_RECORD_IMAGES) and not os.path.isfile(TF_RECORD_MASKS):            
            with tf.python_io.TFRecordWriter(TF_RECORD_IMAGES) as img_writer:
                with tf.python_io.TFRecordWriter(TF_RECORD_MASKS) as ann_writer:
                    for d in tqdm(os.listdir(TRAIN)):
                        for sub_dir in os.listdir(TRAIN + d):
                            for f in os.listdir(TRAIN + d + '/' + sub_dir):
                                f_name = f.split('.dcm')[0]

                                img = pydicom.dcmread(TRAIN + d + '/' + sub_dir + '/' + f)
                                img = img.pixel_array

                                rle = df[df[COLS[0]] == f_name].values[0][1]                
                                
                                mask = []
                                if len(rle) > 3:
                                    mask = rle2mask(rle, img.shape[1], img.shape[0])
                                else:
                                    mask = np.zeros((img.shape[0], img.shape[1]))

                                img = cv2.resize(img, IMG_SIZE) / 255
                                mask = cv2.resize(mask, IMG_SIZE) / 255
                        
                                example = convert_image(img.tostring())
                                img_writer.write(example.SerializeToString())
                            
                                example = convert_image(mask.tostring(), True)
                                ann_writer.write(example.SerializeToString())
        else:
            img_dataset = tf.data.TFRecordDataset([TF_RECORD_IMAGES])
            img_dataset = img_dataset.map(extract_images)
            img_iterator = img_dataset.make_one_shot_iterator()
            next_image_data = img_iterator.get_next()

            ann_dataset = tf.data.TFRecordDataset([TF_RECORD_MASKS])
            ann_dataset = ann_dataset.map(extract_annotations)
            ann_iterator = ann_dataset.make_one_shot_iterator()
            next_ann_data = ann_iterator.get_next()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                try:
                    # Keep extracting data till TFRecord is exhausted
                    while True:
                        image_data = sess.run(next_image_data)
                        ann_data = sess.run(next_ann_data)

                        img = np.reshape(np.frombuffer(image_data), IMG_SIZE)
                        mask = np.reshape(np.frombuffer(ann_data), IMG_SIZE)

                        cv2.imshow('Original', img)
                        cv2.imshow('Mask', mask)

                        key = cv2.waitKey()

                        if key == 27:
                            cv2.destroyAllWindows()
                            break
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
