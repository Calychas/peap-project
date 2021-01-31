import os

import click
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.optimizer_v2.adam import Adam
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout, Input
import numpy as np
from src.models.fasttext_research import get_train_val_test_dataframes
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


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


def run_multilayer_perceptron_experiments(train_tweets, val_tweets, train_polemo, train_wordnet, path_to_results: str):
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
                    f1 = tfa.metrics.F1Score(4, 'macro')
                    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=[f1])

                    history = model.fit(x=np.array(list(train_df['embeddings'])),
                                        y=np.array(list(train_df['label_enc'])),
                                        validation_data=(np.array(list(val_tweets['embeddings'])),
                                                         np.array(list(val_tweets['label_enc']))),
                                        batch_size=500, epochs=100,
                                        sample_weight=np.array(list(train_df['weight'])), verbose=0)

                    results = pd.DataFrame(data={
                        "epoch": range(1, 101),
                        "train_tweets_f_score": history.history['f1_score'],
                        "val_tweets_f_score": history.history['val_f1_score'],
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
                    print(history.history['f1_score'][-1])
                    print(history.history['val_f1_score'][-1])
                    print(history.history['loss'][-1])
                    print(history.history['val_loss'][-1])


def run_best_models(train_tweets, val_tweets, train_polemo, val_polemo, train_wordnet, val_wordnet, test_tweets,
             path_to_results: str, encoder, path_to_plots: str):
    best_parameters = {"first": (
    model_2, 0.0001, 10, (train_polemo, train_tweets, train_wordnet, val_polemo, val_tweets, val_wordnet)),
                       "second": (model_2, 0.0001, 15,
                                  (train_polemo, train_tweets, train_wordnet, val_polemo, val_tweets, val_wordnet)),
                       "third": (model_2, 0.005, 10, (train_polemo, train_tweets, val_polemo, val_tweets)),
                       "fourth": (model_2, 0.00005, 10, (train_polemo, train_tweets, val_polemo, val_tweets)),
                       "fifth": (model_2, 0.00005, 15, (train_polemo, train_tweets, val_polemo, val_tweets)),
                       "sixth": (model_3, 0.005, 10, (train_polemo, train_tweets, val_polemo, val_tweets)),
                       "seventh": (model_3, 0.005, 5, (train_wordnet, train_tweets, val_wordnet, val_tweets)),
                       "eighth": (model_3, 0.00005, 15, (train_polemo, train_tweets, val_polemo, val_tweets))}
    for name, parameters in best_parameters.items():
        train_polemo['weight'] = 1
        train_wordnet['weight'] = 1
        train_tweets['weight'] = parameters[2]
        val_polemo['weight'] = 1
        val_wordnet['weight'] = 1
        val_tweets['weight'] = parameters[2]
        train_df = pd.concat(objs=parameters[3], axis=0)

        model = parameters[0]()
        f1 = tfa.metrics.F1Score(4, 'macro')
        model.compile(optimizer=Adam(learning_rate=parameters[1]), loss='categorical_crossentropy', metrics=[f1])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_f1_score", patience=100)
        history = model.fit(x=np.array(list(train_df['embeddings'])),
                            y=np.array(list(train_df['label_enc'])),
                            validation_data=(np.array(list(test_tweets['embeddings'])),
                                             np.array(list(test_tweets['label_enc']))),
                            batch_size=500, epochs=500, callbacks=[early_stopping],
                            sample_weight=np.array(list(train_df['weight'])), verbose=0)

        results = pd.DataFrame(data={
            "epoch": range(1, len(history.history['f1_score']) + 1),
            "train_tweets_f_score": history.history['f1_score'],
            "test_tweets_f_scores": history.history['val_f1_score'],
            "train_loss": history.history['loss'],
            "val_loss": history.history['val_loss'],
        })
        results.to_csv(
            os.path.join(path_to_results, f"best_{name}.csv"),
            index=False)
        print(name)
        print(history.history['f1_score'][-1])
        print(history.history['val_f1_score'][-1])
        print(history.history['loss'][-1])
        print(history.history['val_loss'][-1])

        labels = ['ambiguous', 'negative', 'neutral', 'positive']
        cm = confusion_matrix(encoder.inverse_transform(np.array(list(test_tweets['label_enc']))),
                              encoder.inverse_transform(model.predict(np.array(list(test_tweets['embeddings'])))),
                              labels)
        df_cm = pd.DataFrame(cm, index=labels,
                             columns=labels)
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(path_to_plots, f"heatmap_{name}.png"))
        plt.close()




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
    "-r",
    "--path-to-results",
    "path_to_results",
    type=click.Path(dir_okay=True, exists=True),
    required=True
)
@click.option(
    "-p",
    "--path-to-plots",
    "path_to_plots",
    type=click.Path(dir_okay=True, exists=True),
    required=True
)
def train_models(train_polemo_embedded: str, val_polemo_embedded: str, test_polemo_embedded: str,
                 political_tweets_embedded: str, wordnet_embedded: str, path_to_results: str,
                 path_to_plots: str):
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


    run_multilayer_perceptron_experiments(train_tweets, val_tweets, train_polemo, train_wordnet, path_to_results)
    run_best_models(train_tweets, val_tweets, train_polemo, val_polemo, train_wordnet, val_wordnet, test_tweets,
             path_to_results, encoder, path_to_plots)


if __name__ == '__main__':
    train_models()  # pylint: disable=no-value-for-parameter
