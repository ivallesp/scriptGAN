import tensorflow as tf

from src.tf_frankenstein.activations import leaky_relu
from src.tf_frankenstein.decoder import build_lstm_feed_back_layer
from src.tf_frankenstein.normalization import BatchNorm



def encoder(input_, max_length, z_depth, reuse=False):
    with tf.variable_scope("Discriminator_encoder", reuse=reuse):
        bn_z = BatchNorm(name="batch_normalization_z")
        bn_d_1 = BatchNorm(name="batch_normalization_d_1")
        bn_d_2 = BatchNorm(name="batch_normalization_d_2")
        bn_d_3 = BatchNorm(name="batch_normalization_d_2")

        cell = tf.nn.rnn_cell.LSTMCell(1024, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer())
        x,_ = tf.nn.dynamic_rnn(cell, bn_z(input_), dtype=tf.float32, scope="DynamicRNN")
        x = tf.reshape(x, shape=[-1, x.shape[2].value], name="stack_LSTM")
        x = tf.layers.dense(bn_d_1(x), 512, activation=leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense_1")
        x = tf.layers.dense(bn_d_2(x), 16, activation=leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense_2")
        x = tf.reshape(x, shape=[input_.shape[0].value, max_length, 16], name="unstack_LSTM")
        x = tf.reshape(x, shape=[input_.shape[0].value, -1], name="combination_LSTM")
        x = tf.layers.dense(bn_d_3(x), z_depth, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense_o")
    return x


def decoder(z, scope, reuse, max_length, batch_size, vocabulary_size):
    with tf.variable_scope(scope, reuse=reuse):
        bn_z = BatchNorm(name="batch_normalization_z")
        bn_zh = BatchNorm(name="batch_normalization_zh")
        bn_zc = BatchNorm(name="batch_normalization_zc")
        bn_d_1 = BatchNorm(name="batch_normalization_d_1")
        bn_d_2 = BatchNorm(name="batch_normalization_d_2")
        bn_d_3 = BatchNorm(name="batch_normalization_d_3")

        z_norm = bn_z(z)
        zh_projected = tf.layers.dense(z_norm, 1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense_zh_projection")
        zc_projected = tf.layers.dense(z_norm, 1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense_zc_projection")

        output_dec, states_dec = build_lstm_feed_back_layer(bn_zh(zh_projected), bn_zc(zc_projected),
                                                            max_length=max_length, name="gen_feed_back")

        lstm_stacked_output = tf.reshape(output_dec, shape=[-1, output_dec.shape[2].value], name="g_stack_LSTM")
        d = tf.layers.dense(bn_d_1(lstm_stacked_output), 512, activation=leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense_1")
        d = tf.layers.dense(bn_d_2(d), 256, activation=leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense_2")
        d = tf.layers.dense(bn_d_3(d), vocabulary_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense_3")

        unstacked_output = tf.reshape(d, shape=[batch_size, max_length, vocabulary_size], name="g_unstack_LSTM")
        o=tf.nn.softmax(unstacked_output)
    return(o)

def discriminator(input_, z_depth, max_length, batch_size, vocabulary_size, reuse=False):
    code = encoder(input_, max_length, z_depth, reuse=reuse)
    output = decoder(code, scope="Discriminator_decoder", reuse=reuse, max_length=max_length, batch_size=batch_size,
                     vocabulary_size=vocabulary_size)
    return(output)

def generator(z, max_length, batch_size, vocabulary_size, reuse=False):
    output = decoder(z, scope="Generator_decoder", reuse=reuse, max_length=max_length, batch_size=batch_size,
                     vocabulary_size=vocabulary_size)
    return(output)


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))

def calculate_slogan_penalty(real_data, fake_data, err_real_data, err_fake_data, eps=1e-8):
    with tf.variable_scope("slogan_penalty"):
        axes_red = list(range(1, len(real_data.shape)))
        dist = tf.sqrt(tf.reduce_sum((real_data-fake_data)**2, axis=axes_red), "l2_distance")
        lip_estimation = tf.abs(err_real_data-err_fake_data)/(dist+eps)
        lip_penalty = tf.minimum(1-lip_estimation, 0.0)**2
    return lip_penalty


class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class GAN:
    def __init__(self, batch_size=512, noise_depth=100, max_length=64, code_size=100, name="GAN"):
        self.name = name
        self.n_neurons_rnn_gen = 1024
        self.batch_size = batch_size
        self.max_length = max_length
        self.noise_depth = noise_depth
        self.vocabulary_size = code_size
        self.optimizer_generator = tf.train.AdamOptimizer(learning_rate=0.00001)
        self.optimizer_discriminator = tf.train.AdamOptimizer(learning_rate=0.00001)
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
            kt = tf.placeholder(dtype=tf.float32, shape=None, name="kt")
            gamma = tf.placeholder(dtype=tf.float32, shape=None, name="gamma")
            acc_1g = tf.placeholder(dtype=tf.float32, shape=None, name="acc_1g")
            acc_2g = tf.placeholder(dtype=tf.float32, shape=None, name="acc_2g")
            acc_3g = tf.placeholder(dtype=tf.float32, shape=None, name="acc_3g")
            ohe_in = tf.one_hot(codes_in, depth=self.vocabulary_size)

        return {"codes_in": codes_in, "ohe_in": ohe_in, "z": z, "kt": kt, "gamma": gamma, "acc_1g": acc_1g, "acc_2g": acc_2g, "acc_3g": acc_3g}

    def define_core_model(self):
        with tf.variable_scope("Core_Model"):
            G = generator(z=self.placeholders.z,
                          max_length=self.max_length,
                          batch_size=self.batch_size,
                          vocabulary_size=self.vocabulary_size,
                          reuse=False)

            D_fake = discriminator(input_=G,
                                   z_depth=self.placeholders.z.shape[-1].value,
                                   max_length=self.max_length,
                                   batch_size=self.batch_size,
                                   vocabulary_size=self.vocabulary_size,
                                   reuse=False)

            D_real = discriminator(input_=self.placeholders.ohe_in,
                                   z_depth=self.placeholders.z.shape[-1].value,
                                   max_length=self.max_length,
                                   batch_size=self.batch_size,
                                   vocabulary_size=self.vocabulary_size,
                                   reuse=True)
        return {"G": G, "D_real": D_real, "D_fake": D_fake}


    def define_losses(self):
        loss_d_real = l1_loss(self.core_model.D_real, self.placeholders.ohe_in)
        loss_d_fake = l1_loss(self.core_model.D_fake, self.core_model.G)
        loss_d = loss_d_real - self.placeholders.kt * loss_d_fake
        loss_g = loss_d_fake
        return {"loss_d": loss_d, "loss_g": loss_g, "loss_d_real": loss_d_real, "loss_d_fake": loss_d_fake}

    def define_optimizers(self):
        self.g_vars = list(filter(lambda k: "Generator" in k.name, tf.trainable_variables()))
        self.d_vars = list(filter(lambda k: "Discriminator" in k.name, tf.trainable_variables()))
        with tf.variable_scope("Optimizers"):
            g_op = self.optimizer_generator.minimize(self.losses.loss_g, var_list=self.g_vars)
            d_op = self.optimizer_discriminator.minimize(self.losses.loss_d, var_list=self.d_vars)
        return {"G": g_op, "D": d_op}

    def define_summaries(self):
        with tf.variable_scope("Summaries"):
            m_global = self.losses.loss_d_real + tf.abs(self.placeholders.gamma*self.losses.loss_d_real - self.losses.loss_d_fake)
            train_final_scalar_probes = {"D_loss": tf.squeeze(self.losses.loss_d),
                                         "G_loss": tf.squeeze(self.losses.loss_g),
                                         "GAN_Loss": tf.squeeze((self.losses.loss_d + self.losses.loss_g) / 2),
                                         "GAN_Equilibrium": tf.squeeze(self.losses.loss_d - self.losses.loss_g),
                                         "M_Global": m_global}

            final_performance_scalar = [tf.summary.scalar(k, tf.reduce_mean(v), family=self.name)
                                        for k, v in train_final_scalar_probes.items()]

            test_scalar_probes = {"1_gram_accuracy": self.placeholders.acc_1g,
                                  "2_gram_accuracy": self.placeholders.acc_2g,
                                  "3_gram_accuracy": self.placeholders.acc_3g}
            test_performance_scalar = [tf.summary.scalar(k, v, family=self.name) for k, v in test_scalar_probes.items()]

        return {"scalar_final_performance": tf.summary.merge(final_performance_scalar),
                "scalar_test_performance": tf.summary.merge(test_performance_scalar)}


__architectures__ = {"GAN": GAN}
