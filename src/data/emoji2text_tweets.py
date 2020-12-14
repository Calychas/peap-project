import click
import pandas as pd
from tqdm.auto import tqdm
import json


@click.command()
@click.option(
    "-i",
    "--input-file",
    "input_file",
    type=click.Path(dir_okay=False, exists=True, readable=True),
    required=True,
)
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, writable=True),
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
    "-u",
    "--emoji-underscored",
    "emoji_underscored",
    type=bool,
    required=True,
    default=False
)
def emoji2text_tweets(input_file: str, output_file: str, emoji_mapping_file, emoji_underscored: bool):
    tqdm.pandas()

    click.echo(f"\nReading emoji mapping")
    emoji_mapping = json.load(emoji_mapping_file)
    emoji_mapping_items = emoji_mapping.items()

    click.echo(f"Emoji underscored: {emoji_underscored}")

    def emoji2text_tweet(tweet: str) -> str:
        text = tweet
        for emoji, emoji_text in emoji_mapping_items:
            text = text.replace(emoji, f" {emoji_text.replace(' ', '_')} " if emoji_underscored else f"<{emoji_text}>")
        return text

    click.echo(f"Reading tweets from {input_file}")
    df: pd.DataFrame = pd.read_pickle(input_file)

    click.echo("Converting tweets")
    df["tweet"] = df["tweet"].progress_apply(lambda x: emoji2text_tweet(x))

    click.echo(f"Saving tweets to {output_file}")
    df.to_pickle(output_file)


if __name__ == "__main__":
    emoji2text_tweets()  # pylint: disable=no-value-for-parameter
