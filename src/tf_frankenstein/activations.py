import tensorflow as tf

leaky_relu = lambda x: tf.maximum(x, 0.1 * x)