import os
from typing import Tuple

import click
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.optimizer_v2.adam import Adam

import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout, Input
import numpy as np
from src.models.fasttext_research import get_train_val_test_dataframes
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler


def model_1():
    model = Sequential()
    model.add(Input(768))
    model.add(Dense(4, activation='softmax'))
    return model


def model_2():
    model = Sequential()
    model.add(Input(768))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model


def model_3():
    model = Sequential()
    model.add(Input(768))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))
    return model


class MetricsCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_tweets_embeddings, train_tweets_labels,
                 val_tweets_embeddings, val_tweets_labels):
        super(MetricsCallback, self).__init__()
        self.train_tweets_embeddings = train_tweets_embeddings
        self.train_tweets_labels = train_tweets_labels
        self.val_tweets_embeddings = val_tweets_embeddings
        self.val_tweets_labels = val_tweets_labels

    def on_train_begin(self, logs={}):
        self.train_tweets_f_scores = []
        self.val_tweets_f_scores = []

    def on_epoch_end(self, epoch, logs={}):
        train_tweet_fscore = self.calc_and_print_f1_score(self.train_tweets_embeddings, self.train_tweets_labels)
        val_tweet_fscore = self.calc_and_print_f1_score(self.val_tweets_embeddings, self.val_tweets_labels)
        self.train_tweets_f_scores.append(train_tweet_fscore)
        self.val_tweets_f_scores.append(val_tweet_fscore)

        logs['train_tweets_f_scores'] = self.train_tweets_f_scores
        logs['val_tweets_f_scores'] = self.val_tweets_f_scores

    def calc_and_print_f1_score(self, embeddings, labels):
        one_hot_predict = self.model.predict(np.array(list(embeddings)))
        f_score = f1_score(np.argmax(np.array(list(labels)), axis=1), np.argmax(one_hot_predict, axis=1),
                           average='macro')
        return f_score


def run_multilayer_perceptron_experiments(train_tweets, val_tweets, train_polemo, train_wordnet, path_to_results: str):
    metric_callback = MetricsCallback(train_tweets['embeddings'], train_tweets['label_enc'],
                                      val_tweets['embeddings'], val_tweets['label_enc'])
    for m in [model_1, model_2, model_3]:
        for lr in [0.00005, 0.0001, 0.0005, 0.001, 0.005]:
            for name, train_dataset in {"polemo_tweets": (train_polemo, train_tweets),
                                        "wordnet_tweets": (train_wordnet, train_tweets),
                                        "polemo_wordnet_tweets": (train_polemo, train_wordnet, train_tweets)}.items():
                for importance_sampling_weight in [5, 10, 15]:
                    train_polemo['weight'] = 1
                    train_wordnet['weight'] = 1
                    train_tweets['weight'] = importance_sampling_weight
                    train_df = pd.concat(objs=train_dataset, axis=0)

                    model = m()
                    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')

                    history = model.fit(x=np.array(list(train_df['embeddings'])),
                                        y=np.array(list(train_df['label_enc'])),
                                        validation_data=(np.array(list(val_tweets['embeddings'])),
                                                         np.array(list(val_tweets['label_enc']))),
                                        batch_size=500, epochs=100, callbacks=[metric_callback],
                                        sample_weight=np.array(list(train_df['weight'])), verbose=0)

                    results = pd.DataFrame(data={
                        "epoch": range(1, 101),
                        "train_tweets_f_score": history.history['train_tweets_f_scores'][-1],
                        "val_tweets_f_score": history.history['val_tweets_f_scores'][-1],
                        "train_loss": history.history['loss'],
                        "val_loss": history.history['val_loss'],
                    })
                    results.to_csv(
                        os.path.join(path_to_results, f"{m.__name__}_{lr}_{name}_{importance_sampling_weight}.csv"),
                        index=False)
                    print(m.__name__)
                    print(lr)
                    print(name)
                    print(importance_sampling_weight)
                    print(history.history['train_tweets_f_scores'][-1][-1])
                    print(history.history['val_tweets_f_scores'][-1][-1])
                    print(history.history['loss'][-1])
                    print(history.history['val_loss'][-1])

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
    "-tmlp",
    "--train-mlps",
    "train_mlps",
    type=bool,
    required=True,
    default=True
)
@click.option(
    "-r",
    "--path-to-results",
    "path_to_results",
    type=click.Path(dir_okay=True, exists=True),
    required=True
)
def train_models(train_polemo_embedded: str, val_polemo_embedded: str, test_polemo_embedded: str,
                 political_tweets_embedded: str, wordnet_embedded: str, train_mlps: bool, path_to_results: str):
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

    train_tweets, val_tweets, test_tweets = get_train_val_test_dataframes(tweets['embeddings'], tweets['label'], 0.8,
                                                                          0.1, 0.1)
    train_wordnet, val_wordnet, test_wordnet = get_train_val_test_dataframes(wordnet['embeddings'], wordnet['label'],
                                                                             0.8, 0.1, 0.1)

    rename_dict = {"text": "embeddings"}
    train_tweets.rename(columns=rename_dict, inplace=True)
    val_tweets.rename(columns=rename_dict, inplace=True)
    test_tweets.rename(columns=rename_dict, inplace=True)
    train_wordnet.rename(columns=rename_dict, inplace=True)
    val_wordnet.rename(columns=rename_dict, inplace=True)
    test_wordnet.rename(columns=rename_dict, inplace=True)

    encoder = LabelBinarizer()
    train_polemo['label_enc'] = list(encoder.fit_transform(list(train_polemo['label'])))
    val_polemo['label_enc'] = list(encoder.transform(list(val_polemo['label'])))
    test_polemo['label_enc'] = list(encoder.transform(list(test_polemo['label'])))
    train_tweets['label_enc'] = list(encoder.transform(list(train_tweets['label'])))
    val_tweets['label_enc'] = list(encoder.transform(list(val_tweets['label'])))
    test_tweets['label_enc'] = list(encoder.transform(list(test_tweets['label'])))
    train_wordnet['label_enc'] = list(encoder.transform(list(train_wordnet['label'])))
    val_wordnet['label_enc'] = list(encoder.transform(list(val_wordnet['label'])))
    test_wordnet['label_enc'] = list(encoder.transform(list(test_wordnet['label'])))

    if train_mlps:
        run_multilayer_perceptron_experiments(train_tweets, val_tweets, train_polemo, train_wordnet, path_to_results)


if __name__ == '__main__':
    train_models()  # pylint: disable=no-value-for-parameter
