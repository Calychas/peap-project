import click
import pandas as pd
from tqdm.auto import tqdm
import src.data.utils as utils
from typing import Set


def remove_stop_words_from_text(text: str, stop_words: Set[str]) -> str:
    return " ".join(
        list(filter(lambda token: token not in stop_words, text.split(" ")))
    )


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
    "-s",
    "--stop-words-file",
    "stop_words_file",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
)
def remove_stop_words(input_file: str, output_file: str, stop_words_file: str):
    tqdm.pandas()

    click.echo(f"\nReading stopwords from {stop_words_file}")
    stop_words = set(utils.jsonc_load(stop_words_file))

    click.echo(f"Reading tweets from {input_file}")
    df: pd.DataFrame = pd.read_pickle(input_file)

    click.echo("Removing stop words")
    df["tweet"] = df["tweet"].progress_apply(
        lambda text: remove_stop_words_from_text(text, stop_words)
    )

    click.echo(f"Saving tweets to {output_file}")
    df.to_pickle(output_file)


if __name__ == "__main__":
    remove_stop_words()  # pylint: disable=no-value-for-parameter
