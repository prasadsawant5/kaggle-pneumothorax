import argparse
import os
from tensorFlow.train import TfTrain

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--framework", required=True, help="Framework to be used for training, e.g. tf (for TensorFlow) or mx (for mxnet)", default='tf')

    args = vars(ap.parse_args())

    if args['framework'] == 'tf':
        tfTrain = TfTrain()
        tfTrain.run()