from src.data.utils import maybe_create_dir, list_full_paths, get_file_name
import click
import pandas as pd
from os.path import join
from typing import List, Dict, Tuple
from tqdm import tqdm
import json


@click.command()
@click.option(
    "-i",
    "--input-dir",
    "input_dir",
    type=click.Path(file_okay=False, exists=True, readable=True),
    required=True,
)
@click.option(
    "-e",
    "--emoji-mapping",
    "emoji_mapping_file",
    type=click.File(mode="r", encoding="UTF-8"),
    required=True,
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
)
def emoji2text_tweets(input_dir: str, output_dir: str, emoji_mapping_file):
    maybe_create_dir(output_dir)
    emoji_mapping = json.load(emoji_mapping_file)
    politician_files_paths = list_full_paths(input_dir)

    for path in tqdm(politician_files_paths):
        emoji2text_file(path, output_dir, emoji_mapping)



def emoji2text_file(path: str, output_dir: str, emoji_mapping: Dict[str, str]):
    df = pd.read_csv(path)
    username = get_file_name(path)

    emoji_mapping_items = emoji_mapping.items()
    def emoji2text_tweet(tweet: str) -> str:
        text = tweet
        for emoji, emoji_text in emoji_mapping_items:
            text = text.replace(emoji, f"<{emoji_text}>")
        return text

    df["tweet"] = df["tweet"].apply(lambda x: emoji2text_tweet(x))
    df.to_csv(join(output_dir, f"{username}.csv"), index=False)



if __name__ == "__main__":
    emoji2text_tweets()  # pylint: disable=no-value-for-parameter
