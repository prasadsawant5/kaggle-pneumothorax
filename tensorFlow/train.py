import cv2
import numpy as np
import os
from tqdm import tqdm
import pydicom
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mask_functions import *
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from tensorFlow.architectures.my_model import MyModel
from tensorFlow.architectures.unet import UNet

IMG_WIDTH = 384
IMG_HEIGHT = 384

TF_RECORD_IMAGES = os.path.dirname(__file__) + '../../images.tfrecord'
TF_RECORD_MASKS = os.path.dirname(__file__) + '../../masks.tfrecord'

H5_IMAGES = os.path.dirname(__file__) + '../../images.h5'
H5_MASKS = os.path.dirname(__file__) + '../../masks.h5'

TRAIN = os.path.dirname(__file__) + '../../dicom-images-train/'
LABELS = os.path.dirname(__file__) + '../../dicom-images-train/'

f_id = 0

class TfTrain:
    def neural_net_input(self):
        return tf.compat.v1.placeholder(tf.float32, (None, IMG_HEIGHT, IMG_WIDTH, 1), name='X')

    def neural_net_output(self):
        return tf.compat.v1.placeholder(tf.float32, (None, IMG_HEIGHT, IMG_WIDTH, 1), name='Y')

    def neural_net_keep_prob(self):
        return tf.compat.v1.placeholder(tf.float32, None, 'keep_prob')

    def dice_coeff(self, y_true, y_pred, num_classes=2, axis=[-1]):
        smooth = 1e-5
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

        return score

    def dice_loss(self, y_true, y_pred):
        with tf.name_scope('dice_loss'):
            loss = 1 - self.dice_coeff(y_true, y_pred)
            return loss

    def optimize(self, nn_last_layer, correct_label, learning_rate, num_classes=2):  
        # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
        logits_reshaped = tf.reshape(nn_last_layer, (-1, num_classes), name="logits_reshaped")
        correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

        with tf.name_scope('loss'):
            # Calculate distance from actual labels using cross entropy
            cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits_reshaped, labels=correct_label_reshaped)
            # Take mean for total loss
            loss_op = tf.compat.v1.reduce_mean(cross_entropy, name="fcn_loss")

            tf.compat.v1.summary.histogram('loss', loss_op)

            bce_dice_loss = loss_op + self.dice_loss(correct_label, nn_last_layer)
            # bce_dice_loss = tf.compat.v1.reduce_mean(bce_dice_loss, name='bce_dice_loss')

            tf.compat.v1.summary.histogram('bce_dice_loss', bce_dice_loss)

        # The model implements this operation to find the weights/parameters that would yield correct pixel labels
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(bce_dice_loss, name="fcn_train_op")

        with tf.name_scope('overall_accuracy'):
            # Accuracy
            # Predict the value of each pixel (i.e. which pixel represents a background and which represents a human)
            correct_pred = tf.equal(tf.argmax(logits_reshaped, 1), tf.argmax(correct_label_reshaped, 1))
            accuracy = tf.compat.v1.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

            tf.compat.v1.summary.histogram('accuracy', accuracy)

        return logits_reshaped, optimizer, loss_op, bce_dice_loss, accuracy

    def train(self, sess, epochs, batch_size, model_output, train_op,
                cross_entropy_loss, dice_loss, accuracy, input_image,
                correct_label, keep_prob, learning_rate, kp, writer):

        global f_id

        merged = tf.compat.v1.summary.merge_all()

        images = h5py.File(H5_IMAGES, 'r')
        truth = h5py.File(H5_MASKS, 'r')

        img_datasets = images.keys()
        truth_datasets = truth.keys()

        total_datasets = len(images.keys())

        datagen = ImageDataGenerator(rescale=1.0/255.0)
        X_train = []
        Y_labels = []

        for epoch in range(epochs):
            idx = 0

            for dataset_idx in range(0, total_datasets):            
                img = images.get('images_{}'.format(dataset_idx))
                masks = truth.get('ann_{}'.format(dataset_idx))

                for batch_idx in range(0, img.shape[0] // batch_size):
                    if dataset_idx == total_datasets - 1 and batch_idx == 20:
                        batch_images = img[idx : img.shape[0]]
                        batch_mask = masks[idx : masks.shape[0]]
                    else:
                        batch_images = img[idx : idx+batch_size]
                        batch_mask = masks[idx : idx+batch_size]

                    X_train, X_val, y_train, y_val = train_test_split(batch_images, batch_mask, test_size=0.3)     

                    _, summary = sess.run([train_op, merged], feed_dict={input_image: X_train, correct_label: y_train, keep_prob: 0.7})
                    # loss, bce_dice_loss, _, acc, summary = sess.run([cross_entropy_loss, dice_loss, train_op, accuracy, merged],
                    # feed_dict={input_image: batch_images, correct_label: batch_mask, keep_prob: kp})

                    loss, bce_dice_loss = sess.run([cross_entropy_loss, dice_loss], feed_dict={input_image: X_train, correct_label: y_train, keep_prob: 1.0})
                    acc = sess.run(accuracy, feed_dict={input_image: X_val, correct_label: y_val, keep_prob: 1.0})

                    idx += batch_size

                    # if batch_idx % 2 == 0:
                    print("Epoch {:03d} ... Dataset {:02d}/{} ... ".format(epoch + 1, dataset_idx + 1, total_datasets), "Loss = {:.4f}".format(loss), " Dice Loss = {:.4f}".format(bce_dice_loss), " Accuracy = {:.4f}".format(acc))

                    writer.add_summary(summary, epoch)
                idx = 0

                if epoch % 10 == 0 or epoch == epochs - 1:
                    arr = np.empty((1, IMG_HEIGHT, IMG_WIDTH, 1))
                    arr[0] = img[50]
                    mask = sess.run(model_output, feed_dict={input_image: arr})
                    mask = mask[0]
                    
                    m = (((mask - mask.min()) * 255) / (mask.max() - mask.min())).astype(np.uint8)

                    cv2.imwrite(os.path.dirname(__file__) + '../../samples/mask_{}.jpg'.format(epoch), m)

                    if not os.path.isfile(os.path.dirname(__file__) + './samples/original.jpg'):
                        image = (((arr[0] - arr[0].min()) * 255) / (arr[0].max() - arr[0].min())).astype(np.uint8)
                        cv2.imwrite(os.path.dirname(__file__) + '../../samples/original.jpg', image)

                    if not os.path.isfile(os.path.dirname(__file__) + '../../samples/mask_org.jpg'):
                        m = (((masks[50] - masks[50].min()) * 255) / (masks[50].max() - masks[50].min())).astype(np.uint8)
                        cv2.imwrite(os.path.dirname(__file__) + '../../samples/mask_org.jpg', m)


    def run(self):
        tf.compat.v1.reset_default_graph() 

        epoch = 100
        batch_size = 32
        learning_rate = 1e-3
        kp = 0.7

        attempt_no = 2

        X = self.neural_net_input()
        Y = self.neural_net_output()
        keep_prob = self.neural_net_keep_prob()

        myModel = MyModel()
        model_output = myModel.fcn_model(X, keep_prob)
        # logits = tf.compat.v1.identity(logits, 'logits')

        # Returns the output logits, training operation and cost operation to be used
        # - logits_reshaped: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: cross entropy loss
        # - dice_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits_reshaped, train_op, cross_entropy_loss, dice_loss, accuracy = self.optimize(model_output, Y, learning_rate)
    
        with tf.compat.v1.Session() as sess:        
            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())        

            if not os.path.exists(os.path.dirname(__file__) + '../../tf_logs'):
                os.mkdir(os.path.dirname(__file__) + '../../tf_logs')
            
            writer = tf.compat.v1.summary.FileWriter(os.path.dirname(__file__) + '../../tf_logs/{}/'.format(attempt_no), sess.graph)        

            print("Model build successful, starting training")

            # Train the neural network
            self.train(sess, epoch, batch_size, model_output,
                    train_op, cross_entropy_loss, dice_loss, accuracy, X,
                    Y, keep_prob, learning_rate, 
                    kp, writer)

            # writer = tf.summary.FileWriter('./logs_fcn', sess.graph)
            saver = tf.compat.v1.train.Saver()
            if not os.path.exists(os.path.dirname(__file__) + '../../tf_model'):
                os.mkdir(os.path.dirname(__file__) + '../../tf_model')

            if not os.path.exists(os.path.dirname(__file__) + './tf_model/{}'.format(attempt_no)):
                os.mkdir(os.path.dirname(__file__) + '../../tf_model/{}'.format(attempt_no))
            
            save_path = saver.save(sess, os.path.dirname(__file__) + "../../tf_model/{}/".format(attempt_no))
            writer.close()