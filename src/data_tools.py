import os
import random
import numpy as np

from collections import Counter

from src.data_loaders import data_functions
from src.general_utilities import *


def load_preprocessed_data(data_key ,special_codes, seed=655321, **kwargs):
    data_loader = data_functions[data_key.upper()]
    list_of_phrases = data_loader(**kwargs)

    random.seed(seed)
    random.shuffle(list_of_phrases)

    # Build vocabulary
    count_dict = Counter(" ".join(list_of_phrases))
    vocabulary = list(map(lambda x: x[0], count_dict.most_common(len(count_dict))))
    char_dict = dict(zip(vocabulary, range(len(special_codes), len(vocabulary) + len(special_codes))))
    inverse_char_dict = {v: k for k, v in char_dict.items()}
    inverse_char_dict.update({v:k for (k,v) in special_codes.items()})
    return list_of_phrases, char_dict, inverse_char_dict



def get_latent_vectors_generator(BATCH_SIZE, noise_depth):
    while 1:
        z = np.random.randn(BATCH_SIZE, noise_depth)
        yield (z)


def encode_sentences(sentence, code_dict, max_length, special_codes):
    #sentence = sentence[:(max_length - 3)] if len(sentence) > (
    #max_length - 2) else sentence  # Trim sentence if exceeds max length
    code = [special_codes["<START>"]] + list(map(lambda w: code_dict.get(w, special_codes["<UNK>"]), sentence)) + \
           [special_codes["<END>"]]  # Encode and add start and end symbols
    code = code + [special_codes["<UNK>"]] * (max_length - len(code))  # Pad code
    return code


def get_sentences(data_key, max_length):
    special_codes = {
        "<UNK>": 0,
        "<GO>": 1,
        "<START>": 2,
        "<END>": 3
    }

    sentences, char_dict, char_dict_inverse = load_preprocessed_data(data_key=data_key, special_codes=special_codes, max_length=max_length)
    sentences_encoded = np.array(list(map(lambda s: encode_sentences(s, char_dict, max_length, special_codes), sentences)))
    return sentences, sentences_encoded, char_dict, char_dict_inverse


def one_hot(seq_batch, depth, pytorch_format=False):
    ohe = np.eye(depth)[seq_batch]
    if pytorch_format:
        ohe = ohe.transpose([0,2,1])
    return(ohe)