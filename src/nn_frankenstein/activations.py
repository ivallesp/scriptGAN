import tensorflow as tf

def leaky_relu(x, name=None, alpha=0.1):
    return tf.maximum(x, alpha * x, name=name)