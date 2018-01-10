import os
import random
import numpy as np

from collections import Counter

from src.data_loaders import data_functions
from src.general_utilities import *


def load_preprocessed_data(data_key ,special_codes, seed=655321):
    data_loader = data_functions[data_key.upper()]
    list_of_phrases = data_loader()

    random.seed(seed)
    random.shuffle(list_of_phrases)

    # Build vocabulary
    count_dict = Counter(" ".join(list_of_phrases))
    vocabulary = list(map(lambda x: x[0], count_dict.most_common(len(count_dict))))
    char_dict = dict(zip(vocabulary, range(len(special_codes), len(vocabulary) + len(special_codes))))
    inverse_char_dict = {v: k for k, v in char_dict.items()}
    inverse_char_dict.update({v:k for (k,v) in special_codes.items()})
    return list_of_phrases, char_dict, inverse_char_dict


def get_batcher(iterable, vocabulary_dict, batch_size, start_code, unknown_code, end_code, max_length):
    while 1:
        for batch in batching(iterable, n=batch_size):
            if len(batch) < batch_size:
                continue
            batch_codes = np.array(list(
                map(lambda s: encode_sentences(s, vocabulary_dict, start_code, unknown_code, end_code, max_length), batch)))
            yield batch_codes


def get_latent_vectors_generator(BATCH_SIZE, noise_depth):
    while 1:
        z = np.random.randn(BATCH_SIZE, noise_depth)
        zd = np.random.randint(-1, 1, size=[BATCH_SIZE, noise_depth])+0.0
        yield (z, zd)


def encode_sentences(sentence, vocabulary_dict, start_code, unknown_code, end_code, max_length):
    sentence = sentence[:(max_length - 3)] if len(sentence) > (
    max_length - 2) else sentence  # Trim sentence if exceeds max length
    code = [start_code] + list(map(lambda w: vocabulary_dict.get(w, unknown_code), sentence)) + [end_code]  # Encode and add start and end symbols
    code = code + [unknown_code] * (max_length - len(code))  # Pad code
    return code
