import os

import click
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import pandas as pd
import numpy as np
from tqdm import trange

from src.models.fasttext_research import get_train_val_test_dataframes
from sklearn.preprocessing import LabelBinarizer


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(768, 4)

    def forward(self, x):
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def predict(torch_model, data, encoder=None):
    out = torch_model(data)
    y_pred = np.argmax(out, axis=1)
    if encoder is not None:
        b = np.zeros((y_pred.size()[0], 4))
        b[np.arange(y_pred.size()[0]), y_pred] = 1
        y_pred = encoder.inverse_transform(b)
    return y_pred


def calc_f1_score(data_arr, model) -> float:
    all_data = torch.Tensor(np.vstack(data_arr[:, 0]))
    all_labels = np.vstack(data_arr[:, 1])
    y_pred = predict(model, all_data)
    y_true = np.argmax(all_labels, axis=1)
    return f1_score(y_true, y_pred, average='macro')


def run_multilayer_perceptron_experiments(train_tweets, val_tweets, train_polemo, train_wordnet, path_to_results: str):
    for m in [Model1, Model2, Model3]:
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
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    loss = nn.CrossEntropyLoss()

                    train_data = []
                    samples_weights = []
                    for index, row in train_df.iterrows():
                        train_data.append([row['embeddings'], row['label_enc']])
                        samples_weights.append(row['weight'])

                    val_data = []
                    for index, row in val_tweets.iterrows():
                        val_data.append([row['embeddings'], row['label_enc']])

                    full_train_data_arr = np.array(train_data)
                    full_val_data_arr = np.array(val_data)
                    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

                    train_loader = DataLoader(
                        train_data, batch_size=500, num_workers=1, sampler=sampler)

                    history = dict()
                    history['f1_score'] = []
                    history['val_f1_score'] = []

                    t = trange(100,
                               desc=f'Training model={m.__name__}, lr={lr}, dataset={name}, importance sampl={importance_sampling_weight}',
                               leave=True)
                    for epoch in t:
                        for batch_idx, (data, target) in enumerate(train_loader):
                            optimizer.zero_grad()
                            out = model(data)
                            out_loss = loss(out, np.argmax(target, axis=1))
                            out_loss.backward()
                            optimizer.step()

                        with torch.no_grad():
                            history['f1_score'].append(calc_f1_score(full_train_data_arr, model))
                            history['val_f1_score'].append(calc_f1_score(full_val_data_arr, model))

                        t.set_description(
                            f'Training model={m.__name__}, lr={lr}, dataset={name}, importance sampl={importance_sampling_weight}, train F1 score={history["f1_score"][-1]}, val F1 score={history["val_f1_score"][-1]}',
                            refresh=True)

                    results = pd.DataFrame(data={
                        "epoch": range(1, 101),
                        "train_tweets_f_score": history['f1_score'],
                        "val_tweets_f_score": history['val_f1_score'],
                    })
                    results.to_csv(
                        os.path.join(path_to_results, f"{m.__name__}_{lr}_{name}_{importance_sampling_weight}.csv"),
                        index=False)


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.best_model = None

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics, model):
        if self.best is None:
            self.best = metrics
            self.best_model = model
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_model = model
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)

    def get_best_model(self):
        return self.best_model


def run_best_models(train_tweets, val_tweets, train_polemo, val_polemo, train_wordnet, val_wordnet, test_tweets,
                    path_to_results: str, encoder, path_to_plots: str):
    best_parameters = {
        "first": (Model2, 0.0005, 10, (train_polemo, train_tweets, val_polemo, val_tweets)),
        "second": (Model2, 0.0005, 15, (train_polemo, train_tweets, val_polemo, val_tweets)),
        "third": (Model2, 0.0005, 5, (train_polemo, train_tweets, val_polemo, val_tweets)),
        "fourth": (
            Model2, 0.0005, 10, (train_polemo, train_tweets, train_wordnet, val_polemo, val_tweets, val_wordnet)),
        "fifth": (Model2, 0.001, 15, (train_polemo, train_tweets, val_polemo, val_tweets)),
        "sixth": (Model3, 0.001, 15, (train_polemo, train_tweets, val_polemo, val_tweets))}
    for name, parameters in best_parameters.items():
        train_polemo['weight'] = 1
        train_wordnet['weight'] = 1
        train_tweets['weight'] = parameters[2]
        val_polemo['weight'] = 1
        val_wordnet['weight'] = 1
        val_tweets['weight'] = parameters[2]
        train_df = pd.concat(objs=parameters[3], axis=0)

        model = parameters[0]()
        optimizer = optim.Adam(model.parameters(), lr=parameters[1])
        loss = nn.CrossEntropyLoss()
        train_data = []
        samples_weights = []
        for index, row in train_df.iterrows():
            train_data.append([row['embeddings'], row['label_enc']])
            samples_weights.append(row['weight'])

        val_data = []
        for index, row in test_tweets.iterrows():
            val_data.append([row['embeddings'], row['label_enc']])

        full_train_data_arr = np.array(train_data)
        full_val_data_arr = np.array(val_data)
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

        train_loader = DataLoader(
            train_data, batch_size=500, num_workers=1, sampler=sampler)

        history = dict()
        history['f1_score'] = []
        history['val_f1_score'] = []

        es = EarlyStopping(patience=50, mode='max')
        t = trange(500,
                   desc=f'Training model={name}',
                   leave=True)
        for epoch in t:
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                out = model(data)
                out_loss = loss(out, np.argmax(target, axis=1))
                out_loss.backward()
                optimizer.step()

            with torch.no_grad():
                history['f1_score'].append(calc_f1_score(full_train_data_arr, model))
                history['val_f1_score'].append(calc_f1_score(full_val_data_arr, model))

            if es.step(history['val_f1_score'][-1], model):
                model = es.get_best_model()
                break  # early stop criterion is met, we can stop now

            t.set_description(
                f'Training model={name} train F1 score={history["f1_score"][-1]}, val F1 score={history["val_f1_score"][-1]}',
                refresh=True)

        results = pd.DataFrame(data={
            "epoch": range(1, len(history['f1_score']) + 1),
            "train_tweets_f_score": history['f1_score'],
            "test_tweets_f_scores": history['val_f1_score']
        })
        results.to_csv(
            os.path.join(path_to_results, f"best_{name}.csv"),
            index=False)

        labels = ['ambiguous', 'negative', 'neutral', 'positive']
        with torch.no_grad():
            cm = confusion_matrix(encoder.inverse_transform(np.array(list(test_tweets['label_enc']))),
                                  predict(model, torch.Tensor(np.vstack(list(test_tweets['embeddings']))), encoder),
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
    "--political-tweets-embedded",
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

    # run_multilayer_perceptron_experiments(train_tweets, val_tweets, train_polemo, train_wordnet, path_to_results)
    run_best_models(train_tweets, val_tweets, train_polemo, val_polemo, train_wordnet, val_wordnet, test_tweets,
                    path_to_results, encoder, path_to_plots)


if __name__ == '__main__':
    train_models()  # pylint: disable=no-value-for-parameter
