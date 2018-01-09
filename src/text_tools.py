import nltk
from src.general_utilities import flatten
from nltk import ngrams
import numpy as np
from collections import Counter


def recursive_remove_unks(s, unk_symbol="<UNK>"):
    s = s.strip()
    if s.endswith(unk_symbol):
        s = s[:-len(unk_symbol)]
        s = recursive_remove_unks(s)
    return s

def remove_start_and_end_symbols(s, start_symbol="<START>", end_symbol="<END>"):
    s = s.strip()
    if s.endswith(end_symbol):
        s = s[:-len(end_symbol)]
    if s.startswith(start_symbol):
        s = s[len(start_symbol):]
    return s

def clear_generation(s):
    s = recursive_remove_unks(s)
    s = remove_start_and_end_symbols(s)
    return s

def remove_substrings(s, substrings):
    for substr in substrings:
        s.replace(substr, "")
    return s


class TokenEvaluator:
    def __init__(self, n_grams=1):
        self.n_grams = n_grams

    def calculate_ngrams(self, lists_of_tokens):
        if self.n_grams > 1:
            lists_of_grams = list(map(lambda x: list(ngrams(x, self.n_grams)), lists_of_tokens))
        else:
            lists_of_grams = lists_of_tokens
        return lists_of_grams

    def fit(self, list_of_real_sentences):
        lists_of_tokens = list(map(lambda x: nltk.word_tokenize(x, "english"), list_of_real_sentences))
        lists_of_tokens = map(lambda s:[w.lower() for w in s], lists_of_tokens)
        fdist = Counter(flatten(self.calculate_ngrams(lists_of_tokens)))
        items = list(map(lambda x:x[0], filter(lambda x:x[1] >= 2, dict(fdist).items()))) # Remove hapaxes
        self.unique_items = set(items)

    def evaluate(self, list_of_generated_sentences):
        lists_of_tokens = list(map(lambda x: nltk.word_tokenize(x, "english"), list_of_generated_sentences))
        if self.n_grams==1:
            lists_of_tokens = list(map(lambda s:[w.lower() for w in s if len(w)>2], lists_of_tokens)) # only consider tokens >2 characters
        else:
            lists_of_tokens = list(map(lambda s:[w.lower() for w in s], lists_of_tokens))

        lists_of_grams = self.calculate_ngrams(lists_of_tokens)
        lists_of_grams = filter(lambda x:len(x), lists_of_grams) # Remove the empty lists (e.g. bigrams of 1 token)
        accuracies = [np.mean(list(map(lambda w:w in self.unique_items, s))) for s in lists_of_grams]
        return accuracies