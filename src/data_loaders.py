import pandas as pd
import numpy as np

from collections import Counter
from src.common_paths import *


def load_imdb_data():
    df = pd.read_csv(os.path.join(get_data_path(), "imdb.tsv"), sep="\t")
    df = df[df.titleType == "movie"]
    movie_titles = df.primaryTitle.unique().tolist()
    movie_titles = list(filter(lambda x: len(x) <= 70 and len(x) >= 5, movie_titles))
    return movie_titles


def load_tatoeba_data(max_length):
    df = pd.read_csv(os.path.join(get_data_path(), "tatoeba.tsv"), sep="\t", header=None)
    df = df[df[1] == "eng"]
    df.columns = ["sentence_id", "language", "text"]
    df["length"] = df["text"].map(len)
    df = df[(df.length<=(max_length-2))]
    char_fdist = Counter("".join(df.text.tolist()))
    characters_allowed = list(map(lambda x: x[0], char_fdist.most_common(100)))
    characters_not_allowed = np.setdiff1d(list(dict(char_fdist).keys()), characters_allowed)
    df["has_rare_characters"] = df.text.map(lambda x: len(np.intersect1d(list(x), characters_not_allowed))>0)
    sentences = df[~df.has_rare_characters].text.tolist()
    return sentences


data_functions = {
    "IMDB": load_imdb_data,
    "TATOEBA": load_tatoeba_data
}
