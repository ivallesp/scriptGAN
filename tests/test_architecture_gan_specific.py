import unittest
import math
from src.architecture import __architectures__
import numpy as np
import tensorflow as tf


class ArchitectureTester(unittest.TestCase):

    def test_architectures_gradient_checks(self):  # Specific GAM
        for name, architecture in __architectures__.items():
            arch = architecture()
            placeholders = list(filter(lambda x: not x.startswith("_"), dir(arch.placeholders)))
            feed_dict = {}
            for ph in placeholders:
                ph = getattr(arch.ph, ph)
                ph_shape = [1] if ph.shape.ndims is None else ph.shape
                ph_shape = [2 if dim is None else dim for dim in ph_shape]
                if ph.dtype is tf.int32:
                    feed_dict[ph] = np.random.randint(0, 1, size=ph_shape)
                elif ph.dtype is tf.bool:
                    feed_dict[ph] = np.random.choice([False, True], size=ph_shape)
                else:
                    feed_dict[ph] = np.random.randn(*ph_shape)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            previous_losses = np.squeeze(sess.run(arch.losses.loss_g, feed_dict=feed_dict))
            sess.run(arch.optimizers.G, feed_dict=feed_dict)
            current_losses = np.squeeze(sess.run(arch.losses.loss_g, feed_dict=feed_dict))
            self.assertTrue(all(previous_losses > current_losses), "Generator op. didn't pass gradient check!")

            previous_losses = np.squeeze(sess.run(arch.losses.loss_d, feed_dict=feed_dict))
            sess.run(arch.optimizers.D, feed_dict=feed_dict)
            current_losses = np.squeeze(sess.run(arch.losses.loss_d, feed_dict=feed_dict))
            self.assertTrue(all(previous_losses > current_losses), "Discriminator op.didn't pass gradient check!")


tf.reset_default_graph()
tao = tf.get_variable("tao", initializer=1.0)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tao)
sess.run(tf.assign(tao, 2.0))