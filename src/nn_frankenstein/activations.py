import tensorflow as tf

leaky_relu = lambda x: tf.maximum(x, 0.1 * x)


def _sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax(logits, temperature):
    gumbel_softmax_sample = logits + _sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    return y