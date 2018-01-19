import tensorflow as tf


def inception_1d(x, is_train, depth, norm_function, norm_input, activ_function, name):
    """
    Inception 1D module implementation.
    :param x: input to the current module (4D tensor with channels-last)
    :param is_train: it is intented to be a boolean placeholder for controling the BatchNormalization behavior (0D tensor)
    :param depth: linearly controls the depth of the network (int)
    :param norm_function: normalization class (same format as the BatchNorm class above)
    :param norm_input: should the input be normalized or not (bool)
    :param activ_function: tensorflow activation function (e.g. tf.nn.relu)
    :param name: name of the variable scope (str)
    """
    no_norm_function = lambda *args, **kwargs: lambda x, *args, **kwargs: x
    norm_function = no_norm_function if norm_function is None else norm_function
    with tf.variable_scope(name):
        if norm_input:
            x_norm = norm_function(name="norm_input")(x, train=is_train)
        else:
            x_norm = x

        # Branch 1: 64 x conv 1x1
        branch_conv_1_1 = tf.layers.conv1d(inputs=x_norm, filters=16 * depth, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_1_1")
        branch_conv_1_1 = norm_function(name="norm_conv_1_1")(branch_conv_1_1, train=is_train)
        branch_conv_1_1 = activ_function(branch_conv_1_1, "activation_1_1")

        # Branch 2: 128 x conv 3x3
        branch_conv_3_3 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_3_3_1")
        branch_conv_3_3 = norm_function(name="norm_conv_3_3_1")(branch_conv_3_3, train=is_train)
        branch_conv_3_3 = activ_function(branch_conv_3_3, "activation_3_3_1")

        branch_conv_3_3 = tf.layers.conv1d(inputs=branch_conv_3_3, filters=32 * depth, kernel_size=3,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_3_3_2")
        branch_conv_3_3 = norm_function(name="norm_conv_3_3_2")(branch_conv_3_3, train=is_train)
        branch_conv_3_3 = activ_function(branch_conv_3_3, "activation_3_3_2")

        # Branch 3: 128 x conv 5x5
        branch_conv_5_5 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_5_5_1")
        branch_conv_5_5 = norm_function(name="norm_conv_5_5_1")(branch_conv_5_5, train=is_train)
        branch_conv_5_5 = activ_function(branch_conv_5_5, "activation_5_5_1")

        branch_conv_5_5 = tf.layers.conv1d(inputs=branch_conv_5_5, filters=32 * depth, kernel_size=5,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_5_5_2")
        branch_conv_5_5 = norm_function(name="norm_conv_5_5_2")(branch_conv_5_5, train=is_train)
        branch_conv_5_5 = activ_function(branch_conv_5_5, "activation_5_5_2")

        # Branch 4: 128 x conv 7x7
        branch_conv_7_7 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_7_7_1")
        branch_conv_7_7 = norm_function(name="norm_conv_7_7_1")(branch_conv_7_7, train=is_train)
        branch_conv_7_7 = activ_function(branch_conv_7_7, "activation_7_7_1")

        branch_conv_7_7 = tf.layers.conv1d(inputs=branch_conv_7_7, filters=32 * depth, kernel_size=5,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_7_7_2")
        branch_conv_7_7 = norm_function(name="norm_conv_7_7_2")(branch_conv_7_7, train=is_train)
        branch_conv_7_7 = activ_function(branch_conv_7_7, "activation_7_7_2")

        # Branch 5: 16 x (max_pool 3x3 + conv 1x1)
        branch_maxpool_3_3 = tf.layers.max_pooling1d(inputs=x_norm, pool_size=3, strides=1, padding="same",
                                                     name="maxpool_3")
        branch_maxpool_3_3 = norm_function(name="norm_maxpool_3_3")(branch_maxpool_3_3, train=is_train)
        branch_maxpool_3_3 = tf.layers.conv1d(inputs=branch_maxpool_3_3, filters=16, kernel_size=1,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              padding="same", name="conv_maxpool_3")

        # Branch 6: 16 x (max_pool 5x5 + conv 1x1)
        branch_maxpool_5_5 = tf.layers.max_pooling1d(inputs=x_norm, pool_size=5, strides=1, padding="same",
                                                     name="maxpool_5")
        branch_maxpool_5_5 = norm_function(name="norm_maxpool_5_5")(branch_maxpool_5_5, train=is_train)
        branch_maxpool_5_5 = tf.layers.conv1d(inputs=branch_maxpool_5_5, filters=16, kernel_size=1,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              padding="same", name="conv_maxpool_5")

        # Branch 7: 16 x (avg_pool 3x3 + conv 1x1)
        branch_avgpool_3_3 = tf.layers.average_pooling1d(inputs=x_norm, pool_size=3, strides=1, padding="same",
                                                         name="avgpool_3")
        branch_avgpool_3_3 = norm_function(name="norm_avgpool_3_3")(branch_avgpool_3_3, train=is_train)
        branch_avgpool_3_3 = tf.layers.conv1d(inputs=branch_avgpool_3_3, filters=16, kernel_size=1,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              padding="same", name="conv_avgpool_3")

        # Branch 8: 16 x (avg_pool 5x5 + conv 1x1)
        branch_avgpool_5_5 = tf.layers.average_pooling1d(inputs=x_norm, pool_size=5, strides=1, padding="same",
                                                         name="avgpool_5")
        branch_avgpool_5_5 = norm_function(name="norm_avgpool_5_5")(branch_avgpool_5_5, train=is_train)
        branch_avgpool_5_5 = tf.layers.conv1d(inputs=branch_avgpool_5_5, filters=16, kernel_size=1,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              padding="same", name="conv_avgpool_5")

        # Concatenate
        output = tf.concat([branch_conv_1_1, branch_conv_3_3, branch_conv_5_5, branch_conv_7_7, branch_maxpool_3_3,
                            branch_maxpool_5_5, branch_avgpool_3_3, branch_avgpool_5_5], axis=-1)
        return output


def inception_1d_v2(x, is_train, depth, norm_function, norm_input, activ_function, name):
    """
    Inception 1D module implementation.
    :param x: input to the current module (4D tensor with channels-last)
    :param is_train: it is intented to be a boolean placeholder for controling the BatchNormalization behavior (0D tensor)
    :param depth: linearly controls the depth of the network (int)
    :param norm_function: normalization class (same format as the BatchNorm class above)
    :param norm_input: should the input be normalized or not (bool)
    :param activ_function: tensorflow activation function (e.g. tf.nn.relu)
    :param name: name of the variable scope (str)
    """
    no_norm_function = lambda *args, **kwargs: lambda x, *args, **kwargs: x
    norm_function = no_norm_function if norm_function is None else norm_function
    with tf.variable_scope(name):
        if norm_input:
            x_norm = norm_function(name="norm_input")(x, train=is_train)
        else:
            x_norm = x

        # Branch 1: 64 x conv 1x1
        branch_conv_1_1 = tf.layers.conv1d(inputs=x_norm, filters=16 * depth, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_1_1")
        branch_conv_1_1 = norm_function(name="norm_conv_1_1")(branch_conv_1_1, train=is_train)
        branch_conv_1_1 = activ_function(branch_conv_1_1, "activation_1_1")

        # Branch 4: 32*depth x conv 3 x 3
        branch_conv_3_3 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_3_3_1")
        branch_conv_3_3 = norm_function(name="norm_conv_3_3_1")(branch_conv_3_3, train=is_train)
        branch_conv_3_3 = activ_function(branch_conv_3_3, "activation_3_3_1")

        branch_conv_3_3 = tf.layers.conv1d(inputs=branch_conv_3_3, filters=32 * depth, kernel_size=3,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_3_3_2")
        branch_conv_3_3 = norm_function(name="norm_conv_3_3_2")(branch_conv_3_3, train=is_train)
        branch_conv_3_3 = activ_function(branch_conv_3_3, "activation_3_3_2")

        # Branch 4: 32*depth x conv 5 x 5
        branch_conv_5_5 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_5_5_1")
        branch_conv_5_5 = norm_function(name="norm_conv_5_5_1")(branch_conv_5_5, train=is_train)
        branch_conv_5_5 = activ_function(branch_conv_5_5, "activation_5_5_1")

        branch_conv_5_5 = tf.layers.conv1d(inputs=branch_conv_5_5, filters=32 * depth, kernel_size=5,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_5_5_2")
        branch_conv_5_5 = norm_function(name="norm_conv_5_5_2")(branch_conv_5_5, train=is_train)
        branch_conv_5_5 = activ_function(branch_conv_5_5, "activation_5_5_2")

        # Branch 4: 32*depth x conv 7 x 7
        branch_conv_7_7 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_7_7_1")
        branch_conv_7_7 = norm_function(name="norm_conv_7_7_1")(branch_conv_7_7, train=is_train)
        branch_conv_7_7 = activ_function(branch_conv_7_7, "activation_7_7_1")

        branch_conv_7_7 = tf.layers.conv1d(inputs=branch_conv_7_7, filters=32 * depth, kernel_size=7,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_7_7_2")
        branch_conv_7_7 = norm_function(name="norm_conv_7_7_2")(branch_conv_7_7, train=is_train)
        branch_conv_7_7 = activ_function(branch_conv_7_7, "activation_7_7_2")

        # Branch 4: 32*depth x conv 9 x 9
        branch_conv_9_9 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_9_9_1")
        branch_conv_9_9 = norm_function(name="norm_conv_9_9_1")(branch_conv_9_9, train=is_train)
        branch_conv_9_9 = activ_function(branch_conv_9_9, "activation_9_9_1")

        branch_conv_9_9 = tf.layers.conv1d(inputs=branch_conv_9_9, filters=32 * depth, kernel_size=9,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_9_9_2")
        branch_conv_9_9 = norm_function(name="norm_conv_9_9_2")(branch_conv_9_9, train=is_train)
        branch_conv_9_9 = activ_function(branch_conv_9_9, "activation_9_9_2")

        # Branch 4: 32*depth x conv 11 x 11
        branch_conv_11_11 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_11_11_1")
        branch_conv_11_11 = norm_function(name="norm_conv_11_11_1")(branch_conv_11_11, train=is_train)
        branch_conv_11_11 = activ_function(branch_conv_11_11, "activation_11_11_1")

        branch_conv_11_11 = tf.layers.conv1d(inputs=branch_conv_11_11, filters=32 * depth, kernel_size=11,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_11_11_2")
        branch_conv_11_11 = norm_function(name="norm_conv_11_11_2")(branch_conv_11_11, train=is_train)
        branch_conv_11_11 = activ_function(branch_conv_11_11, "activation_11_11_2")

        # Branch 4: 32*depth x conv 13 x 13
        branch_conv_13_13 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_13_13_1")
        branch_conv_13_13 = norm_function(name="norm_conv_13_13_1")(branch_conv_13_13, train=is_train)
        branch_conv_13_13 = activ_function(branch_conv_13_13, "activation_13_13_1")

        branch_conv_13_13 = tf.layers.conv1d(inputs=branch_conv_13_13, filters=32 * depth, kernel_size=13,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_13_13_2")
        branch_conv_13_13 = norm_function(name="norm_conv_13_13_2")(branch_conv_13_13, train=is_train)
        branch_conv_13_13 = activ_function(branch_conv_13_13, "activation_13_13_2")

        # Branch 4: 32*depth x conv 15 x 15
        branch_conv_15_15 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_15_15_1")
        branch_conv_15_15 = norm_function(name="norm_conv_15_15_1")(branch_conv_15_15, train=is_train)
        branch_conv_15_15 = activ_function(branch_conv_15_15, "activation_15_15_1")

        branch_conv_15_15 = tf.layers.conv1d(inputs=branch_conv_15_15, filters=32 * depth, kernel_size=15,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_15_15_2")
        branch_conv_15_15 = norm_function(name="norm_conv_15_15_2")(branch_conv_15_15, train=is_train)
        branch_conv_15_15 = activ_function(branch_conv_15_15, "activation_15_15_2")

        
        
	# Concatenate
        output = tf.concat([branch_conv_1_1, branch_conv_3_3, branch_conv_5_5, branch_conv_7_7, 
                            branch_conv_9_9, branch_conv_11_11, branch_conv_13_13, branch_conv_15_15], axis=-1)
        return output
