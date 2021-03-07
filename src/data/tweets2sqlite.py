import pandas as pd
from sqlalchemy import create_engine
import click


@click.command()
@click.option(
    "-t",
    "--tweets-file",
    "tweets_file",
    type=click.Path(dir_okay=False, exists=True, readable=True),
    required=True,
)
@click.option(
    "-s",
    "--sqlite-file",
    "sqlite_file",
    type=click.Path(dir_okay=False, readable=True),
    required=True,
)
@click.option(
    "-n",
    "--db-name",
    "db_name",
    type=click.STRING,
    default='tweets',
    required=True
)
def save_to_sqlite(tweets_file: str, sqlite_file: str, db_name: str):
    engine = create_engine(f'sqlite:///{sqlite_file}')
    tweets = pd.read_pickle(tweets_file)
    tweets = tweets.drop(columns=['tweet'])
    tweets.to_sql(db_name, con=engine)


if __name__ == '__main__':
    save_to_sqlite()  # pylint: disable=no-value-for-parameter
