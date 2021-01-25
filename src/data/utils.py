from os.path import exists, join
from os import mkdir, listdir
from pathlib import Path
import regex as re
import json
from typing import Set, Dict, List
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


def emoji2text_tweet(tweet: str, emoji_mapping_items: Dict[str, str]) -> str:
    text = tweet
    for emoji, emoji_text in emoji_mapping_items:
        text = text.replace(emoji, f"<{emoji_text}>")
    return text


def read_embeddings_dataframe(df_path: str) -> pd.DataFrame:
    embeddings_column_name = "embedding"
    csv_df = pd.read_csv(df_path)
    embeddings = np.array(
        [
            np.array([np.float(i) for i in x.replace("]", "").replace("[", "").split()])
            for x in csv_df[embeddings_column_name].tolist()
        ]
    )
    correct_df = csv_df.drop(columns=[embeddings_column_name])
    correct_df[embeddings_column_name] = embeddings
    return correct_df
