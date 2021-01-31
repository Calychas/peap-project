import click
import pandas as pd
from keras.layers import Input, Dense
from keras import Sequential
from sklearn.preprocessing import LabelBinarizer
import tensorflow_addons as tfa
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

def neural_network_model():
    model = Sequential()
    model.add(Input(768))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model


def train_model_and_save(dataset, path_to_model: str):
    model = neural_network_model()
    f1 = tfa.metrics.F1Score(4, 'macro')
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=[f1])
    model.fit(x=np.array(list(dataset['embeddings'])),
                        y=np.array(list(dataset['label_enc'])),
                        batch_size=500, epochs=200, sample_weight=np.array(list(dataset['weight'])))

    model.save(path_to_model)




@click.command()
@click.option(
    "-trp",
    "--train-polemo-embedded",
    "train_polemo_embedded",
    type=click.Path(file_okay=True, exists=True),
    required=True
)
@click.option(
    "-vp",
    "--val-polemo-embedded",
    "val_polemo_embedded",
    type=click.Path(file_okay=True, exists=True),
    required=True
)
@click.option(
    "-tp",
    "--test-polemo-embedded",
    "test_polemo_embedded",
    type=click.Path(file_okay=True, exists=True),
    required=True
)
@click.option(
    "-pt",
    "--political-tweets-tembedded",
    "political_tweets_embedded",
    type=click.Path(file_okay=True, exists=True),
    required=True
)
@click.option(
    "-w",
    "--wordnet-embedded",
    "wordnet_embedded",
    type=click.Path(file_okay=True, exists=True),
    required=True,
)
@click.option(
    "-o",
    "--output",
    "output",
    type=click.Path(file_okay=True, exists=False),
    required=True,
)
def train_models(train_polemo_embedded: str, val_polemo_embedded: str, test_polemo_embedded: str,
                 political_tweets_embedded: str, wordnet_embedded: str, output: str):
    train_polemo = pd.read_pickle(train_polemo_embedded)
    val_polemo = pd.read_pickle(val_polemo_embedded)
    test_polemo = pd.read_pickle(test_polemo_embedded)
    tweets = pd.read_pickle(political_tweets_embedded)
    wordnet = pd.read_pickle(wordnet_embedded)

    train_polemo = train_polemo[['embeddings', 'label']]
    val_polemo = val_polemo[['embeddings', 'label']]
    test_polemo = test_polemo[['embeddings', 'label']]
    tweets = tweets[['embeddings', 'label']]
    wordnet = wordnet[['embeddings', 'label']]

    encoder = LabelBinarizer()
    train_polemo['label_enc'] = list(encoder.fit_transform(list(train_polemo['label'])))
    val_polemo['label_enc'] = list(encoder.transform(list(val_polemo['label'])))
    test_polemo['label_enc'] = list(encoder.transform(list(test_polemo['label'])))
    tweets['label_enc'] = list(encoder.transform(list(tweets['label'])))
    wordnet['label_enc'] = list(encoder.transform(list(wordnet['label'])))

    train_polemo['weight'] = 1
    val_polemo['weight'] = 1
    test_polemo['weight'] = 1
    tweets['weight'] = 10
    wordnet['weight'] = 1
    train_df = pd.concat(objs=(train_polemo, val_polemo, test_polemo, tweets, wordnet), axis=0)
    train_model_and_save(train_df, output)


if __name__ == '__main__':
    train_models()  # pylint: disable=no-value-for-parameter
