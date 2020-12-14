import click
import pandas as pd
from tqdm.auto import tqdm
import regex as re


def process_text(text):
    text = re.sub(r"http\S+", "", text)  # removes http links
    text = re.sub(r"\S+\.com\S+", "", text)  # removes links that have no http but end with com
    text = re.sub(r"\S+\.pl\S+", "", text)  # removes links that have no http but end with pl
    text = re.sub(r"\@\w+", "", text)  # removes whole mentions
    text = re.sub(r"\#", "", text)  # removes hashes (content of hashtag remains)
    text = re.sub(r"\s+", " ", text)  # convert multiple spaces into one
    return text


@click.command()
@click.option(
    "-i",
    "--input-path",
    "input_path",
    type=click.Path(dir_okay=False, exists=True, readable=True),
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    "output_path",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
)
@click.option(
    "-l",
    "--min-tweet-length",
    "min_tweet_length",
    type=int,
    default=10,
    required=True,
)
@click.option(
    "-c",
    "--min-tweets-count",
    "min_tweets_count",
    type=int,
    default=20,
    required=True,
)
@click.option(
    "-p",
    "--leave-only-polish",
    "leave_only_polish",
    type=bool,
    default=True,
    required=True,
)
def clean_tweets(
    input_path: str,
    output_path: str,
    min_tweet_length: int,
    min_tweets_count: int,
    leave_only_polish: bool,
):
    tqdm.pandas()
    click.echo(f"\nReading from {input_path}")
    df: pd.DataFrame = pd.read_pickle(input_path)

    if leave_only_polish:
        click.echo("Leaving only polish tweets")
        df = df[df["language"] == "pl"]

    # CLEAN TWEETS
    click.echo("Cleaning tweets")
    df["tweet"] = df["tweet"].progress_apply(process_text)

    # DROP BY TWEET LENGTH
    click.echo(f"Dropping tweets with length < {min_tweet_length}")
    df["tweet_length"] = df["tweet"].progress_apply(len)
    df = df[df["tweet_length"] >= min_tweet_length]
    df.drop(columns="tweet_length")

    # DROP BY TWEETS COUNT
    click.echo(f"Dropping usernames with tweet count < {min_tweets_count}")
    tweets_by_username_count = df.groupby(by="username")[["id"]].count()
    users_to_drop = (
        tweets_by_username_count[tweets_by_username_count["id"] < min_tweets_count]
        .reset_index()["username"]
        .to_list()
    )
    df: pd.DataFrame = df[~df["username"].isin(users_to_drop)]  # type: ignore

    click.echo(f"Saving to {output_path}")
    df.to_pickle(output_path)


if __name__ == "__main__":
    clean_tweets()  # pylint: disable=no-value-for-parameter
