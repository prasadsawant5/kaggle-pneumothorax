from mxnet import gluon
import mxnet as mx
from mxnet.gluon import nn

class MyModel(nn.HybridSequential):
    def fcn_conv2d_bn(self, encoder, filters, conv_ksize, keep_prob, conv_strides=2, pool_ksize=2, pool_strides=2, padding=(1,1), activation='relu', is_batch_norm=True, is_training=True, is_dropout=False):
        with encoder.name_scope():
            encoder.add(nn.Conv2D(channels=filters, kernel_size=conv_ksize, strides=conv_strides, padding=padding, layout='NHWC', activation=activation))

            if is_batch_norm:
                encoder.add(nn.BatchNorm(axis=3))

            if is_dropout:
                encoder.add(nn.Dropout(kp))

    def fcn_conv_2d_transpose_bn(self, decoder, filters, conv_ksize, keep_prob, conv_strides=2, padding=(1,1), activation='relu', is_batch_norm=True, is_training=True, is_dropout=False):
        with decoder.name_scope():
            decoder.add(nn.Conv2DTranspose(channels=filters, kernel_size=conv_ksize, strides=conv_strides, padding=padding, layout='NHWC', activation=activation))

            if is_batch_norm:
                decoder.add(nn.BatchNorm(axis=3))

            if is_dropout:
                decoder.add(nn.Dropout(kp))

    def fcn_output(self, decoder, filters, ksize, strides=2, padding=(1,1), activation='sigmoid'):
        with decoder.name_scope():
            decoder.add(nn.Conv2DTranspose(channels=1, kernel_size=ksize, strides=strides, padding=padding, layout='NHWC', activation=activation))

    def __init__(self, **kwards):
        nn.HybridSequential.__init__(self, **kwards)

        self.encoder = nn.HybridSequential()
        self.fcn_conv2d_bn(self.encoder, 32, 3, keep_prob=0.7)
        self.fcn_conv2d_bn(self.encoder, 64, 3, keep_prob=0.7)
        self.fcn_conv2d_bn(self.encoder, 128, 3, keep_prob=0.7)
        self.fcn_conv2d_bn(self.encoder, 128, 3, keep_prob=0.7, conv_strides=1)

        self.decoder = nn.HybridSequential()
        self.fcn_conv_2d_transpose_bn(self.decoder, 64, 3, keep_prob=0.7)
        self.fcn_conv_2d_transpose_bn(self.decoder, 32, 3, keep_prob=0.7)
        self.fcn_output(self.decoder, 1, 3)

    def hybrid_forward(self, F, x):
        fcn0_32 = self.encoder[0](x)
        fcn1_64 = self.encoder[1](fcn0_32)
        fcn2_128 = self.encoder[2](fcn1_64)
        mid3 = self.encoder[3](fcn2_128)

        fcn4_64 = self.decoder[0](mid3)
        fcn5_skip = F.concat([fcn4_64, fcn1_64])
        fcn6_32 = self.decoder[1](fcn5_skip)
        fcn7_skip = F.concat([fcn6_32, fcn0_32])
        out = self.decoder[2](fcn7_skip)

        logits = F.Softmax(out, axis=1)

        return logits