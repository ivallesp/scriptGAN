# coding: utf-8

import codecs

import tensorflow as tf

from src.architecture_embedding import GAN
from src.common_paths import *
from src.data_tools import load_preprocessed_data, get_batcher, get_latent_vectors_generator
from src.general_utilities import get_exponential_generator
from src.tensorflow_utilities import start_tensorflow_session, get_summary_writer, TensorFlowSaver
from src.text_tools import *

# Load configuration
config = json.load(open("settings.json"))

# Load data and add tags to dictionary
special_codes = {
    "<UNK>": 0,
    "<GO>": 1,
    "<START>": 2,
    "<END>": 3
}
sentences, char_dict, char_dict_inverse = load_preprocessed_data(data_key="TATOEBA", special_codes=special_codes)
charset_cardinality = len(char_dict_inverse)

# Define parameters
project_id = "GAN_TATOEBA"
version_id = "V12"
logs_path = get_tensorboard_logs_path()
BATCH_SIZE = 32
critic_its = 10
noise_depth = 100
batches_test = 10
test_period = 100
save_period = 5000
max_std_noise, min_std_noise = [0.2, 0.005]
max_length = np.max(list(map(len, sentences)))

it = 0

te_1 = TokenEvaluator(n_grams=1)
te_2 = TokenEvaluator(n_grams=2)
te_3 = TokenEvaluator(n_grams=3)
te_1.fit(list_of_real_sentences=sentences)
te_2.fit(list_of_real_sentences=sentences)
te_3.fit(list_of_real_sentences=sentences)


gan = GAN(batch_size=BATCH_SIZE, noise_depth=noise_depth, max_length=max_length, vocabulary_size=charset_cardinality)

sess = start_tensorflow_session(device=str(config["device"]), memory_fraction=config["memory_fraction"])
sw = get_summary_writer(sess, logs_path, project_id, version_id)
saver = TensorFlowSaver(path=os.path.join(get_model_path(project_id, version_id), "model"))
sess.run(tf.global_variables_initializer())

# In[9]:

# Define generators

batch_gen = get_batcher(sentences, char_dict, batch_size=BATCH_SIZE,
                        start_code=special_codes["<START>"],
                        unknown_code=special_codes["<UNK>"],
                        end_code=special_codes["<END>"],
                        max_length=max_length)

latent_batch_gen = get_latent_vectors_generator(BATCH_SIZE, noise_depth)

# In[6]:
std_noise_gen = get_exponential_generator(max_std_noise, min_std_noise, eta=0.00002)
# Define operations
while 1:
    std_noise = next(std_noise_gen)
    for _ in range(critic_its):
        batch = next(batch_gen)
        z = next(latent_batch_gen)
        sess.run(gan.op.D, feed_dict={gan.ph.codes_in: batch, gan.ph.z: z, gan.ph.std_dev_codes: (std_noise,)})

    batch = next(batch_gen)
    z = next(latent_batch_gen)

    sess.run(gan.op.G, feed_dict={gan.ph.codes_in: batch, gan.ph.z: z, gan.ph.std_dev_codes: (std_noise,)})

    if (it % test_period) == 0:  # Reporting...
        generation=[]
        for bt in range(batches_test):
            batch = next(batch_gen)
            z = next(latent_batch_gen)
            s, generation_code = sess.run([gan.summ.scalar_final_performance, gan.core_model.G],
                                          feed_dict={gan.ph.codes_in: batch,
                                                     gan.ph.z: z,
                                          gan.ph.std_dev_codes: (std_noise,)})

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



