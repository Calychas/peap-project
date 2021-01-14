import pickle as pkl

import click
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@click.command()
@click.option(
    "-t",
    "--topics-count",
    "topics_count",
    type=click.INT,
    required=True
)
@click.option(
    "-i",
    "--input",
    "input_tweets_file",
    type=click.Path(dir_okay=False, readable=True),
    required=True
)
@click.option(
    "-v",
    "--vectorizer",
    "output_vectorizer_file",
    type=click.Path(dir_okay=False, writable=True),
    required=True
)
@click.option(
    "-m",
    "--model",
    "output_model_file",
    type=click.Path(dir_okay=False, writable=True),
    required=True
)
def train_topic_model(
        topics_count: int,
        input_tweets_file: str,
        output_vectorizer_file: str,
        output_model_file: str
):
    df = pd.read_pickle(input_tweets_file)

    tweets = df.tweet.to_list()

    vectorizer = CountVectorizer(min_df=2, max_df=0.95, max_features=4000)
    counts = vectorizer.fit_transform(tweets)

    with open(output_vectorizer_file, "wb") as vec_file:
        pkl.dump(vectorizer, vec_file, protocol=pkl.HIGHEST_PROTOCOL)

    click.echo(f"Saved vectorizer at {output_vectorizer_file}")

    lda = LatentDirichletAllocation(n_components=topics_count, random_state=42,
                                    verbose=True, evaluate_every=10, n_jobs=15,
                                    max_iter=100)
    lda.fit(counts)

    with open(output_model_file, "wb") as model_file:
        pkl.dump(lda, model_file, protocol=pkl.HIGHEST_PROTOCOL)

    click.echo(f"Saved LDA model at {output_model_file}")


if __name__ == '__main__':
    train_topic_model()  # pylint: disable=no-value-for-parameter
