import tensorflow as tf

class UNet:
    def unet_conv2d_bn(self, x, filters, conv_ksize, keep_prob, scope_name, conv_strides=1, pool_ksize=2, pool_strides=2, padding='same', activation='relu', is_batch_norm=True, is_training=False, is_dropout=False, is_max_pool=True):
        k_init = tf.random_normal_initializer()

        with tf.name_scope(scope_name):
            conv = tf.compat.v1.layers.conv2d(x, filters=filters, kernel_size=conv_ksize, strides=conv_strides, padding=padding, activation=activation, kernel_initializer=k_init)
            conv = tf.compat.v1.layers.conv2d(conv, filters=filters, kernel_size=conv_ksize, strides=conv_strides, padding=padding, activation=activation, kernel_initializer=k_init)

            normalized = None

            if is_batch_norm:
                normalized = tf.compat.v1.layers.batch_normalization(conv, training=is_training)

            if is_max_pool:
                normalized = tf.compat.v1.layers.max_pooling2d(conv, pool_size=pool_ksize, strides=pool_strides)
            else:
                normalized = conv

            if is_dropout:
                normalized = tf.nn.dropout(conv, keep_prob=keep_prob)

            return conv, normalized

    def unet_conv2d_transpose_bn(self, x, filters, ksize, keep_prob, scope_name, conv_layer, strides=2, padding='same', activation='relu'):
        k_init = tf.random_normal_initializer()

        with tf.name_scope(scope_name):
            conv_transpose = tf.compat.v1.layers.conv2d_transpose(x, filters=filters, kernel_size=ksize, strides=strides, padding=padding, activation=activation, kernel_initializer=k_init)
            conv_transpose = tf.compat.v1.concat([conv_transpose, conv_layer], axis=-1)

            conv_transpose = tf.compat.v1.layers.conv2d(conv_transpose, filters=filters, kernel_size=3, strides=1, padding=padding, activation=activation, kernel_initializer=k_init)
            conv_transpose = tf.compat.v1.layers.conv2d(conv_transpose, filters=filters, kernel_size=3, strides=1, padding=padding, activation=activation, kernel_initializer=k_init)

            return conv_transpose

    def unet_output(self, x, filters, ksize, strides=1, padding='valid', activation='sigmoid', name='conv_transpose_unet_output', scope_name='output'):
        k_init = tf.random_normal_initializer()

        with tf.name_scope(scope_name):
            output = tf.compat.v1.layers.conv2d(x, filters=filters, kernel_size=ksize, strides=strides, padding=padding, activation=activation, kernel_initializer=k_init, name=name)

            return output

    def unet_model(self, X, kp, is_training=True, num_classes=2):
        conv0, fcn0 = self.unet_conv2d_bn(X, 16, 3, keep_prob=kp, scope_name='conv0')

        conv1, fcn1 = self.unet_conv2d_bn(fcn0, 32, 3, keep_prob=kp, scope_name='conv1')

        conv2, fcn2 = self.unet_conv2d_bn(fcn1, 64, 3, keep_prob=kp, scope_name='conv2')

        conv3, fcn3 = self.unet_conv2d_bn(fcn2, 128, 3, keep_prob=kp, scope_name='conv3')

        conv4, fcn4 = self.unet_conv2d_bn(fcn3, 256, 3, keep_prob=kp, scope_name='conv4', is_max_pool=False)

        fcn5 = self.unet_conv2d_transpose_bn(fcn4, 128, 3, keep_prob=kp, scope_name='conv_transpose0', conv_layer=conv3)

        fcn6 = self.unet_conv2d_transpose_bn(fcn5, 64, 3, keep_prob=kp, scope_name='conv_transpose1', conv_layer=conv2)

        fcn7 = self.unet_conv2d_transpose_bn(fcn6, 32, 3, keep_prob=kp, scope_name='conv_transpose2', conv_layer=conv1)

        fcn8 = self.unet_conv2d_transpose_bn(fcn7, 16, 3, keep_prob=kp, scope_name='conv_transpose3', conv_layer=conv0)

        output = self.unet_output(fcn8, 1, 1)

        return output



