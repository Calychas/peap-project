import click
import pandas as pd
from tqdm import tqdm
import src.data.utils as utils


@click.command()
@click.option(
    "-i",
    "--input-dir",
    "input_dir",
    type=click.Path(file_okay=False, exists=True, readable=True),
    required=True,
)
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
)
def csv2pkl(input_dir: str, output_file: str):
    full_paths = utils.list_full_paths(input_dir)
    df = pd.concat(map(lambda x: pd.read_csv(x), full_paths))
    df.to_pickle(output_file)


if __name__ == "__main__":
    csv2pkl()  # pylint: disable=no-value-for-parameter
