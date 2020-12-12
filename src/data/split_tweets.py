from pandas.io.parsers import read_csv
from src.data.utils import maybe_create_dir, list_full_paths, get_file_name
import click
import pandas as pd
from os import listdir
from os.path import join
from typing import List
from tqdm import tqdm
import multiprocessing as mp

@click.command("split")
@click.option(
    "-i",
    "--input-dir",
    "input_dir",
    type=click.Path(file_okay=False, exists=True, readable=True),
    required=True,
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
)
def split_tweets(input_dir: str, output_dir: str):
    maybe_create_dir(output_dir)
    politician_files_paths = list_full_paths(input_dir)
    split_politician_files(politician_files_paths, output_dir)


def split_politician_files(paths: List[str], output_dir: str):
    for path in tqdm(paths):
        split_csv_file(path, output_dir)


def split_csv_file(path: str, output_dir: str):
    df = pd.read_csv(path)
    tweets = get_tweets(df)
    username = get_file_name(path)
    save_tweets(tweets, output_dir, username)


def get_tweets(df: pd.DataFrame) -> List[str]:
    return df.tweet.to_list()


def save_tweets(tweets: List[str], output_dir: str, username: str):
    for i, tweet in enumerate(tweets):
        file_path = join(output_dir, f"{username}_{i}.txt")
        save_tweet(file_path, tweet)


def save_tweet(file_path: str, tweet: str):
    with open(file_path, "wt", encoding="utf-8") as f:
        f.write(tweet)


if __name__ == "__main__":
    split_tweets()
