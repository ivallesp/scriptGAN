# coding: utf-8
import codecs

from src.architecture import GAN
from src.common_paths import *
from src.data_tools import get_latent_vectors_generator, get_sentences
from src.general_utilities import *
from src.text_tools import *
from src.data_tools import one_hot

import torch
import torch.optim as optim
import torch.functional as F
import torch.nn as nn
import torch.autograd as autograd

import numpy as np
from src.pytorch_utilities import use_devices, get_summary_writer


# Define parameters
project_id = "GAN_TATOEBA"
version_id = "V35"
logs_path = get_tensorboard_logs_path()
batch_size = 64
critic_its = 10
noise_depth = 100
batches_test = 10
test_period = 100
save_period = 5000
restore = False
max_length = 64
cuda=True
it=0

# Load configuration
use_devices(2)


# Load data
sentences, sentences_encoded, char_dict, char_dict_inverse = get_sentences("TATOEBA", max_length=max_length)
sentences_onehot = one_hot(sentences_encoded, len(char_dict_inverse), pytorch_format=True)
codes_batch_gen = batching(list_of_iterables=[sentences_onehot, sentences],
                           n=batch_size,
                           infinite=True,
                           return_incomplete_batches=False)

cardinality = len(char_dict_inverse)

# Define generators
latent_batch_gen = get_latent_vectors_generator(batch_size, noise_depth)


charset_cardinality = len(char_dict_inverse)

te_1 = TokenEvaluator(n_grams=1)
te_2 = TokenEvaluator(n_grams=2)
te_3 = TokenEvaluator(n_grams=3)
te_1.fit(list_of_real_sentences=sentences)
te_2.fit(list_of_real_sentences=sentences)
te_3.fit(list_of_real_sentences=sentences)

# Initialize architecture
gan = GAN(noise_depth=noise_depth, batch_size=batch_size, n_outputs=cardinality, max_length=max_length)

sw = get_summary_writer(logs_path=logs_path, project_id=project_id, version_id=version_id, remove_if_exists=True)

# Define operations
dtype=torch.cuda.FloatTensor if cuda else torch.FloatTensor
while 1:
    for _ in range(critic_its):
        real_data = autograd.Variable(torch.from_numpy(next(codes_batch_gen)[0])).type(dtype)
        z = autograd.Variable(torch.from_numpy(next(latent_batch_gen))).type(dtype)
        cost_d = gan.op.D(real_data, z)

    z = autograd.Variable(torch.from_numpy(next(latent_batch_gen))).type(dtype)

    cost_g = gan.op.G(z)

    if (it % test_period) == 0:
        generation=[]
        d_loss = g_loss = w_approx = 0
        for bt in range(batches_test):
            real_data = autograd.Variable(torch.from_numpy(next(codes_batch_gen)[0])).type(dtype)
            z = autograd.Variable(torch.from_numpy(next(latent_batch_gen))).type(dtype)
            generation_code = gan.core_model.G.forward(z).data.cpu().numpy()
            d_loss += gan.losses.D(gan.core_model.G, gan.core_model.D, real_data, z)
            g_loss += gan.losses.G(gan.core_model.G, gan.core_model.D, z)
            w_approx += gan.losses.W(gan.core_model.D, real_data, z)

            generation.extend(list(map(
                lambda x: "".join(list(map(lambda c: char_dict_inverse.get(c, "<ERROR>"),
                                           np.argmax(x, axis=0).tolist()))), generation_code)))

        filepath = os.path.join(get_output_path(project_id, version_id), "gen_{0:08d}.txt".format(it))
        with codecs.open(filepath, "w", "utf-8") as f:
            f.write("\r\n".join(generation))

        generation = list(map(recursive_remove_unks, generation))
        generation_clean = list(map(lambda x:remove_substrings(x, list(["<UNK>", "<GO>", "<START>", "<END>"])), generation))
        acc_1g = te_1.evaluate(generation_clean)
        acc_2g = te_2.evaluate(generation_clean)
        acc_3g = te_3.evaluate(generation_clean)

        print("{1}\n=== Iteration {0} | 1-gram acc: {2:.5f} | 2-gram acc: {3:.5f} | 3-gram acc: {4:.5f} ===\n".format(it,
                                                                     "\n".join(generation[0:20]),
                                                                     np.mean(acc_1g), np.mean(acc_2g), np.mean(acc_3g)))
        gan.summ.acc_1(sw, np.mean(acc_1g), it)
        gan.summ.acc_2(sw, np.mean(acc_2g), it)
        gan.summ.acc_3(sw, np.mean(acc_3g), it)
        gan.summ.loss_summaries(sw, g_loss/batches_test, d_loss/batches_test, w_approx/batches_test, it)

        if (it % 5000) == 0:  # Saving...
            pass #Save

    it += 1
