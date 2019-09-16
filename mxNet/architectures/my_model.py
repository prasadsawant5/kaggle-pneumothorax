from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn

class MyMode():
    def fcn_conv2d_bn(self, encoder, filters, conv_ksize, keep_prob, conv_strides=2, pool_ksize=2, pool_strides=2, padding=(1,1), activation='relu', is_batch_norm=True, is_training=True, is_dropout=False):
        with encoder.name_scope():
            encoder.add(nn.Conv2D(channels=filters, kernel_size=conv_ksize, strides=conv_strides, padding=padding, layout='NHWC', activation=activation))

            if is_batch_norm:
                encoder.add(nn.BatchNorm(axis=3))

            if is_drop_out:
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

        

    def fcn_model(self, x, kp, is_training=True, num_classes=2):
        net = nn.HybridSequential()
        self.fcn_conv2d_bn(net, 32, 3, kp)
        self.fcn_conv2d_bn(net, 64, 3, kp)
        self.fcn_conv2d_bn(net, 128, 3, kp)
        self.fcn_conv2d_bn(net, 128, 3, kp, conv_strides=1)

        self.fcn_conv_2d_transpose_bn(net, 64, 3, kp)
        net.add(net[4] + net[1])
        self.fcn_conv_2d_transpose_bn(net, 32, 3)
        net.add(net[5], net[0])
        self.fcn_output(net, 1, 3)

        net.initialize(init=init.Xavier())

        return net