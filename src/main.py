# coding: utf-8
import codecs

import tensorflow as tf

from src.architecture import GAN
from src.common_paths import *
from src.data_tools import get_latent_vectors_generator, get_sentences
from src.general_utilities import *
from src.tensorflow_utilities import start_tensorflow_session, get_summary_writer, TensorFlowSaver
from src.text_tools import *


# Define parameters
project_id = "GAN_TATOEBA"
version_id = "V36"
logs_path = get_tensorboard_logs_path()
batch_size = 512
critic_its = 10
noise_depth = 100
batches_test = 10
test_period = 100
save_period = 5000
restore = False
max_length = 64


# Load configuration
config = json.load(open("settings.json"))

# Load data
sentences, sentences_encoded, char_dict, char_dict_inverse = get_sentences("TATOEBA", max_length=max_length)
codes_batch_gen = batching(list_of_iterables=[sentences_encoded, sentences],
                           n=batch_size,
                           infinite=True,
                           return_incomplete_batches=False)

charset_cardinality = len(char_dict_inverse)

te_1 = TokenEvaluator(n_grams=1)
te_2 = TokenEvaluator(n_grams=2)
te_3 = TokenEvaluator(n_grams=3)
te_1.fit(list_of_real_sentences=sentences)
te_2.fit(list_of_real_sentences=sentences)
te_3.fit(list_of_real_sentences=sentences)

# Initialize architecture
gan = GAN(batch_size=batch_size, noise_depth=noise_depth, max_length=max_length, code_size=charset_cardinality)

# Restore weights if specified and initialize a saver
sess = start_tensorflow_session(device=str(config["device"]), memory_fraction=config["memory_fraction"])
if restore:
    sw = get_summary_writer(sess, logs_path, project_id, version_id, remove_if_exists=False)
    it = int(sorted(os.listdir(get_output_path(project_id, version_id)))[-1][4:-4])
    saver_restore = tf.train.Saver()
    last_checkpoint = tf.train.latest_checkpoint(get_model_path(project_id, version_id))
    saver_restore.restore(sess, last_checkpoint)
    print("Model {} restored successfully, continuing from iteration {}".format(last_checkpoint, it))
else:
    sw = get_summary_writer(sess, logs_path, project_id, version_id, remove_if_exists=True)
    it = 0
    sess.run(tf.global_variables_initializer())

saver = TensorFlowSaver(path=os.path.join(get_model_path(project_id, version_id), "model"))
####


# Define generators
latent_batch_gen = get_latent_vectors_generator(batch_size, noise_depth)


# Define operations
while 1:
    for _ in range(critic_its):
        batch, _ = next(codes_batch_gen)
        z = next(latent_batch_gen)
        sess.run(gan.op.D, feed_dict={gan.ph.codes_in: batch, gan.ph.z: z})

    batch, _ = next(codes_batch_gen)
    z = next(latent_batch_gen)

    sess.run(gan.op.G, feed_dict={gan.ph.codes_in: batch, gan.ph.z: z})

    if (it % test_period) == 0:  # Reporting...
        generation=[]
        for bt in range(batches_test):
            batch, _ = next(codes_batch_gen)
            z = next(latent_batch_gen)
            s, generation_code = sess.run([gan.summ.scalar_final_performance, gan.core_model.G],
                                          feed_dict={gan.ph.codes_in: batch,
                                                     gan.ph.z: z})

            generation.extend(list(map(
                lambda x: "".join(list(map(lambda c: char_dict_inverse.get(c, "<ERROR>"),
                                           np.argmax(x, axis=1).tolist()))), generation_code)))

        filepath = os.path.join(get_output_path(project_id, version_id), "gen_{0:08d}.txt".format(it))
        with codecs.open(filepath, "w", "utf-8") as f:
            f.write("\r\n".join(generation))

        generation = list(map(recursive_remove_unks, generation))
        generation_clean = list(map(lambda x:remove_substrings(x, list(special_codes.keys())), generation))
        acc_1g = te_1.evaluate(generation_clean)
        acc_2g = te_2.evaluate(generation_clean)
        acc_3g = te_3.evaluate(generation_clean)

        print("{1}\n=== Iteration {0} | 1-gram acc: {2:.5f} | 2-gram acc: {3:.5f} | 3-gram acc: {4:.5f} ===\n".format(it,
                                                                     "\n".join(generation[0:20]),
                                                                     np.mean(acc_1g), np.mean(acc_2g), np.mean(acc_3g)))
        st = sess.run(gan.summ.scalar_test_performance, feed_dict={gan.ph.acc_1g: np.mean(acc_1g),
                                                                   gan.ph.acc_2g: np.mean(acc_2g),
                                                                   gan.ph.acc_3g: np.mean(acc_3g)})


        sw.add_summary(s, it)
        sw.add_summary(st, it)
    if (it % 5000) == 0:  # Saving...
        saver.save(sess, it)

    it += 1



