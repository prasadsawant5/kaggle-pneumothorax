import cv2
import numpy as np
import os
from tqdm import tqdm
import pydicom
import tensorflow as tf
from mask_functions import *
import pandas as pd
import sys
import h5py

IMG_WIDTH = 384
IMG_HEIGHT = 384

TF_RECORD_IMAGES = './images.tfrecord'
TF_RECORD_MASKS = './masks.tfrecord'

H5_IMAGES = './images.h5'
H5_MASKS = './masks.h5'

TRAIN = './dicom-images-train/'

f_id = 0

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

def neural_net_input():
    return tf.placeholder(tf.float32, (None, IMG_HEIGHT, IMG_WIDTH, 1), name='X')

def neural_net_output():
    return tf.placeholder(tf.float32, (None, IMG_HEIGHT, IMG_WIDTH, 1), name='Y')

def neural_net_keep_prob():
    return tf.placeholder(tf.float32, None, 'keep_prob')

def conv2d_bn(x, filters, conv_ksize, keep_prob, scope_name, conv_strides=2, pool_ksize=2, pool_strides=2, padding='same', activation='relu', is_batch_norm=True, is_training=True, is_dropout=False):
    k_init = tf.random_normal_initializer(stddev=0.02)

    with tf.name_scope(scope_name):
        conv = tf.layers.conv2d(x, filters=filters, kernel_size=conv_ksize, strides=conv_strides, padding=padding, activation=activation, kernel_initializer=k_init)

        if is_batch_norm:
            conv = tf.layers.batch_normalization(conv, training=is_training)

        if is_dropout:
            conv = tf.nn.dropout(conv, keep_prob=keep_prob)

        return conv

def conv_2d_transpose_bn(x, filters, ksize, keep_prob, scope_name, strides=2, padding='same', activation='relu', is_batch_norm=True, is_training=True, is_dropout=False):
    k_init = tf.random_normal_initializer(stddev=0.02)

    with tf.name_scope(scope_name):
        conv_transpose = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=ksize, strides=strides, padding=padding, activation=activation, kernel_initializer=k_init)

        if is_batch_norm:
            conv_transpose = tf.layers.batch_normalization(conv_transpose, training=is_training)

        if is_dropout:
            conv_transpose = tf.nn.dropout(conv_transpose, keep_prob)

        return conv_transpose

def skip_conn(x, y, scope_name):
    with tf.name_scope(scope_name):
        x = tf.add(x, y)
        return x

def output(x, filters, ksize, strides=2, padding='same', activation='sigmoid', name='conv_transpose_output', scope_name='output'):
    k_init = tf.random_normal_initializer(stddev=0.02)

    with tf.name_scope(scope_name):
        output = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=ksize, strides=strides, padding=padding, activation=activation, kernel_initializer=k_init, name=name)

        return output

def optimize(nn_last_layer, correct_label, learning_rate, num_classes=2):  
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    tf.summary.histogram('loss', loss_op)

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    # Accuracy
    # Predict the value of each pixel (i.e. which pixel represents a background and which represents a human)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label_reshaped, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    tf.summary.histogram('accuracy', accuracy)

    return logits, train_op, loss_op, accuracy

def model(x, kp, is_training=True, num_classes=2):
    fcn0 = conv2d_bn(x, 25, 32, keep_prob=kp, scope_name='conv0', is_dropout=True)
    print('FCN0: {}'.format(fcn0.get_shape().as_list()))

    fcn1 = conv2d_bn(fcn0, 50, 64, keep_prob=kp, scope_name='conv1', is_dropout=True)
    print('FCN1: {}'.format(fcn1.get_shape().as_list()))

    fcn2 = conv2d_bn(fcn1, 100, 128, keep_prob=kp, scope_name='conv2', is_dropout=True)
    print('FCN2: {}'.format(fcn2.get_shape().as_list()))

    mid = conv2d_bn(fcn2, 100, 128, keep_prob=kp, scope_name='1x1conv', conv_strides=1, is_dropout=True)
    print('MID: {}'.format(mid.get_shape().as_list()))

    fcn3 = conv_2d_transpose_bn(mid, 50, 64, keep_prob=kp, scope_name='conv_transpose0', is_dropout=True)
    print('FCN3: {}'.format(fcn3.get_shape().as_list()))

    fcn3 = skip_conn(fcn3, fcn1, 'skip_conn0')
    print('FCN3_SKIP: {}'.format(fcn3.get_shape().as_list()))

    fcn4 = conv_2d_transpose_bn(fcn3, 25, 32, keep_prob=kp, scope_name='conv_transpose1', is_dropout=True)
    print('FCN4: {}'.format(fcn4.get_shape().as_list()))

    fcn4 = skip_conn(fcn4, fcn0, 'skip_conn1')
    print('FCN4_SKIP: {}'.format(fcn4.get_shape().as_list()))

    op = output(fcn4, 1, 3)
    print('OUTPUT: {}'.format(op.get_shape().as_list()))

    return op

def train(sess, epochs, batch_size, model_output, train_op,
             cross_entropy_loss, accuracy, input_image,
             correct_label, keep_prob, learning_rate, kp, writer):

    global f_id

    merged = tf.summary.merge_all()

    images = h5py.File(H5_IMAGES, 'r')
    truth = h5py.File(H5_MASKS, 'r')

    img_datasets = images.keys()
    truth_datasets = truth.keys()

    total_datasets = len(images.keys())

    for epoch in range(epochs):
        total_loss = 0
        idx = 0

        for dataset_idx in range(0, total_datasets):            
            img = images.get('images_{}'.format(dataset_idx))
            masks = truth.get('ann_{}'.format(dataset_idx))

            for batch_idx in range(0, img.shape[0] // batch_size):
                if dataset_idx == total_datasets - 1 and batch_idx == 13:
                    batch_images = img[idx : img.shape[0]]
                    batch_mask = masks[idx : masks.shape[0]]
                else:
                    batch_images = img[idx : idx+batch_size]
                    batch_mask = masks[idx : idx+batch_size]        

                loss, _, acc, summary = sess.run([cross_entropy_loss, train_op, accuracy, merged],
                feed_dict={input_image: batch_images, correct_label: batch_mask, keep_prob: kp})

                total_loss += loss
                idx += batch_size

                # if batch_idx % 2 == 0:
                print("Epoch {} ... Dataset {}/{} ... ".format(epoch + 1, dataset_idx + 1, total_datasets), "Loss = {:.4f}".format(loss), " Accuracy = {:.4f}".format(acc))
            idx = 0


def run():
    epoch = 10
    batch_size = 32
    learning_rate = 1e-3
    kp = 0.7
  
    with tf.Session() as sess:        
        X = neural_net_input()
        Y = neural_net_output()
        keep_prob = neural_net_keep_prob()

        model_output = model(X, kp)

        writer = tf.summary.FileWriter('./logs/1/', sess.graph)

        # Returns the output logits, training operation and cost operation to be used
        # - logits: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits, train_op, cross_entropy_loss, accuracy = optimize(model_output, Y, learning_rate)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())                

        print("Model build successful, starting training")

        # Train the neural network
        train(sess, epoch, batch_size, model_output,
                train_op, cross_entropy_loss, accuracy, X,
                Y, keep_prob, learning_rate, 
                kp, writer)

        # writer = tf.summary.FileWriter('./logs_fcn', sess.graph)
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./fcn_{}/model".format(epoch))
        writer.close()


if __name__ == '__main__':
    run()