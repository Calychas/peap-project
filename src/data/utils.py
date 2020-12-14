from os.path import exists, join
from os import mkdir, listdir
from typing import List
from pathlib import Path
import regex as re
import json
from typing import Set, Dict
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np


def maybe_create_dir(dir: str):
    if not exists(dir):
        mkdir(dir)


def list_full_paths(dir: str) -> List[str]:
    files = listdir(dir)
    paths = [join(dir, file) for file in files]
    return paths


def get_file_name(path: str) -> str:
    return Path(path).stem


def jsonc_load(path: str):
    text = open(path, "r", encoding="utf-8").read()
    return json.loads(re.sub("//.*", "", text, flags=re.MULTILINE))


def remove_stop_words(frequencies: Dict[str, int], stop_words: Set[str]):
    return dict(filter(lambda x: x[0] not in stop_words, frequencies.items()))


def get_frequencies(series: pd.Series, stop_words: Set[str] = None) -> Dict[str, int]:
    count_vect = CountVectorizer(min_df=2, max_df=1.0, max_features=1000)
    counts = count_vect.fit_transform(series)

    word_counts = np.asarray(counts.sum(axis=0)).squeeze()  # type: ignore
    feature_names = count_vect.get_feature_names()

    frequencies = {name: count for name, count in zip(feature_names, word_counts)}

    if stop_words:
        frequencies = remove_stop_words(frequencies, stop_words)

    frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
    return frequencies