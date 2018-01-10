import tensorflow as tf

from src.nn_frankenstein.activations import leaky_relu
from src.nn_frankenstein.embedding_decoder import build_lstm_feed_back_layer
from src.nn_frankenstein.normalization import BatchNorm


def build_discriminator(input_, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        x = tf.layers.conv1d(input_, filters=64, kernel_size=9, padding="same", strides=1, activation=leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1d_1_1")

        x = tf.layers.conv1d(x, filters=64, kernel_size=9, padding="same", strides=1, activation=leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1d_1_2")
        print(x.shape)
        x = tf.layers.average_pooling1d(x, pool_size=2, strides=2, name="pooling_1")

        x = tf.layers.conv1d(x, filters=128, kernel_size=9, padding="same", strides=1, activation=leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1d_2_1")

        x = tf.layers.conv1d(x, filters=128, kernel_size=9, padding="same", strides=1, activation=leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1d_2_2")
        print(x.shape)
        x = tf.layers.average_pooling1d(x, pool_size=2, strides=2, name="pooling_2")

        x = tf.layers.conv1d(x, filters=256, kernel_size=9, padding="same", strides=1, activation=leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1d_3_1")

        x = tf.layers.conv1d(x, filters=256, kernel_size=9, padding="same", strides=1, activation=leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1d_3_2")
        print(x.shape)
        x = tf.layers.average_pooling1d(x, pool_size=2, strides=2, name="pooling_3")

        x = tf.layers.conv1d(x, filters=512, kernel_size=9, padding="same", strides=1, activation=leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1d_4_1")

        x = tf.layers.conv1d(x, filters=512, kernel_size=9, padding="same", strides=1, activation=leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1d_4_2")
        print(x.shape)
        x = tf.layers.average_pooling1d(x, pool_size=8, strides=2, name="pooling_4")

        x = tf.layers.conv1d(x, filters=1, kernel_size=1, strides=1, activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1d_5_final")
        print(x.shape)
    return x


def build_generator(z, n_neurons_rnn, max_length, vocabulary_size):
    with tf.variable_scope("Generator", reuse=False):
        bn_zh = BatchNorm(name="batch_normalization_zh")
        bn_zc = BatchNorm(name="batch_normalization_zc")

        z_norm = z
        zh_projected = tf.layers.dense(z_norm, n_neurons_rnn, activation=None,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="dense_zh_projection")
        zc_projected = tf.layers.dense(z_norm, n_neurons_rnn, activation=None,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="dense_zc_projection")

        output_dec, states_dec = build_lstm_feed_back_layer(bn_zh(zh_projected), bn_zc(zc_projected),
                                                            max_length=max_length, cardinality_input=vocabulary_size,
                                                            name="gen_feed_back")

    return(output_dec)



class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class GAN:
    def __init__(self, batch_size, noise_depth, max_length, vocabulary_size, name="GAN"):
        self.name = name
        self.n_neurons_rnn_gen = 1024
        self.batch_size = batch_size
        self.max_length = max_length
        self.noise_depth = noise_depth
        self.vocabulary_size = vocabulary_size
        self.optimizer_generator = tf.train.AdamOptimizer(learning_rate=0.00001)
        self.optimizer_discriminator = tf.train.AdamOptimizer(learning_rate=0.00015)
        self.define_computation_graph()

        # Aliases
        self.ph = self.placeholders
        self.op = self.optimizers
        self.summ = self.summaries

    def define_computation_graph(self):
        # Reset graph
        tf.reset_default_graph()
        self.placeholders = NameSpacer(**self.define_placeholders())
        self.core_model = NameSpacer(**self.define_core_model())
        self.losses = NameSpacer(**self.define_losses())
        self.optimizers = NameSpacer(**self.define_optimizers())
        self.summaries = NameSpacer(**self.define_summaries())

    def define_placeholders(self):
        with tf.variable_scope("Placeholders"):
            codes_in = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.max_length), name="codes_in")
            z = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.noise_depth], name="z")
            acc_1g = tf.placeholder(dtype=tf.float32, shape=None, name="acc_1g")
            acc_2g = tf.placeholder(dtype=tf.float32, shape=None, name="acc_2g")
            acc_3g = tf.placeholder(dtype=tf.float32, shape=None, name="acc_3g")
        return {"codes_in": codes_in, "z": z, "acc_1g": acc_1g, "acc_2g": acc_2g, "acc_3g": acc_3g}

    def define_core_model(self):
        with tf.variable_scope("Core_Model"):
            tweet = tf.one_hot(self.placeholders.codes_in, depth=self.vocabulary_size)
            G = build_generator(z=self.placeholders.z, n_neurons_rnn=self.n_neurons_rnn_gen,
                                max_length=self.max_length,
                                vocabulary_size=self.vocabulary_size)
            D_real = build_discriminator(input_=tweet)
            D_fake = build_discriminator(input_=G, reuse=True)
            epsilon = tf.random_uniform(shape=tweet[:, :, 0:1].shape, minval=0., maxval=1.)
            interp = (epsilon) * G + (1 - epsilon) * tweet
            D_interpolates = build_discriminator(input_=interp, reuse=True)

            grad_interpolated = tf.gradients(D_interpolates, [interp])[0]
        return {"G": G, "D_real": D_real, "D_fake": D_fake, "grad_interpolated": grad_interpolated}

    def define_losses(self):
        grads_l2 = tf.sqrt(tf.reduce_sum(tf.square(self.core_model.grad_interpolated),
                                         reduction_indices=[1, 2], keep_dims=True))  # Norm 2
        gradient_penalty = (grads_l2 - 1) ** 2
        with tf.variable_scope("Losses"):
            loss_d_real = self.core_model.D_real
            loss_d_fake = - self.core_model.D_fake
            loss_d = loss_d_fake + loss_d_real + 10 * gradient_penalty
            loss_g = self.core_model.D_fake
        return {"loss_d_real": loss_d_real, "loss_d_fale": loss_d_fake, "loss_d": loss_d, "loss_g": loss_g}

    def define_optimizers(self):
        self.g_vars = list(filter(lambda k: "Generator" in k.name, tf.trainable_variables()))
        self.d_vars = list(filter(lambda k: "Discriminator" in k.name, tf.trainable_variables()))
        with tf.variable_scope("Optimizers"):
            g_op = self.optimizer_generator.minimize(self.losses.loss_g, var_list=self.g_vars)
            d_op = self.optimizer_discriminator.minimize(self.losses.loss_d, var_list=self.d_vars)
        return {"G": g_op, "D": d_op}

    def define_summaries(self):
        with tf.variable_scope("Summaries"):
            train_final_scalar_probes = {"D_loss": tf.squeeze(self.losses.loss_d),
                                         "G_loss": tf.squeeze(self.losses.loss_g),
                                         "GAN_Loss": tf.squeeze((self.losses.loss_d + self.losses.loss_g) / 2),
                                         "GAN_Equilibrium": tf.squeeze(self.losses.loss_d - self.losses.loss_g)}

            final_performance_scalar = [tf.summary.scalar(k, tf.reduce_mean(v), family=self.name)
                                        for k, v in train_final_scalar_probes.items()]

            test_scalar_probes = {"1_gram_accuracy": self.placeholders.acc_1g,
                                  "2_gram_accuracy": self.placeholders.acc_2g,
                                  "3_gram_accuracy": self.placeholders.acc_3g}
            test_performance_scalar = [tf.summary.scalar(k, v, family=self.name) for k, v in test_scalar_probes.items()]

        return {"scalar_final_performance": tf.summary.merge(final_performance_scalar),
                "scalar_test_performance": tf.summary.merge(test_performance_scalar)}

