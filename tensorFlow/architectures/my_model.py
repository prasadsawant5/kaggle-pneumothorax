import tensorflow as tf

class MyModel:
    def __init__(self):
        pass
    
    def fcn_conv2d_bn(self, x, filters, conv_ksize, keep_prob, scope_name, conv_strides=2, pool_ksize=2, pool_strides=2, padding='same', activation='relu', is_batch_norm=True, is_training=True, is_dropout=False):
        k_init = tf.random_normal_initializer()

        with tf.name_scope(scope_name):
            conv = tf.layers.conv2d(x, filters=filters, kernel_size=conv_ksize, strides=conv_strides, padding=padding, activation=activation, kernel_initializer=k_init)

            if is_batch_norm:
                conv = tf.layers.batch_normalization(conv, training=is_training)

            if is_dropout:
                conv = tf.nn.dropout(conv, keep_prob=keep_prob)

            return conv

    def fcn_conv_2d_transpose_bn(self, x, filters, ksize, keep_prob, scope_name, strides=2, padding='same', activation='relu', is_batch_norm=True, is_training=True, is_dropout=False):
        k_init = tf.random_normal_initializer()

        with tf.name_scope(scope_name):
            conv_transpose = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=ksize, strides=strides, padding=padding, activation=activation, kernel_initializer=k_init)

            if is_batch_norm:
                conv_transpose = tf.layers.batch_normalization(conv_transpose, training=is_training)

            if is_dropout:
                conv_transpose = tf.nn.dropout(conv_transpose, keep_prob)

            return conv_transpose

    def fcn_skip_conn(self, x, y, scope_name):
        with tf.name_scope(scope_name):
            x = tf.compat.v1.concat([x, y], axis=-1)
            return x

    def fcn_output(self, x, filters, ksize, strides=2, padding='same', activation='sigmoid', name='conv_transpose_fcn_output', scope_name='fcn_output'):
        k_init = tf.random_normal_initializer()

        with tf.name_scope(scope_name):
            output = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=ksize, strides=strides, padding=padding, activation=activation, kernel_initializer=k_init, name=name)

            return output

    def fcn_model(self, x, kp, is_training=True, num_classes=2):
        fcn0 = self.fcn_conv2d_bn(x, 32, 3, keep_prob=kp, scope_name='conv0', is_dropout=False)

        fcn1 = self.fcn_conv2d_bn(fcn0, 64, 3, keep_prob=kp, scope_name='conv1', is_dropout=False)

        fcn2 = self.fcn_conv2d_bn(fcn1, 128, 3, keep_prob=kp, scope_name='conv2', is_dropout=False)

        mid = self.fcn_conv2d_bn(fcn2, 128, 3, keep_prob=kp, scope_name='1x1conv', conv_strides=1, is_dropout=False)

        fcn3 = self.fcn_conv_2d_transpose_bn(mid, 64, 3, keep_prob=kp, scope_name='conv_transpose0', is_dropout=False)

        fcn3 = self.fcn_skip_conn(fcn3, fcn1, 'fcn_skip_conn0')

        fcn4 = self.fcn_conv_2d_transpose_bn(fcn3, 32, 3, keep_prob=kp, scope_name='conv_transpose1', is_dropout=False)

        fcn4 = self.fcn_skip_conn(fcn4, fcn0, 'fcn_skip_conn1')

        op = self.fcn_output(fcn4, 1, 3)

        return op