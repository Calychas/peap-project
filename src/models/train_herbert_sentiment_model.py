import click
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle as pkl
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import f1_score


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_model_and_save(dataset, path_to_model: str):
    model = NeuralNetwork()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss = nn.CrossEntropyLoss()

    train_data = []
    samples_weights = []
    for index, row in dataset.iterrows():
        train_data.append([row['embeddings'], row['label_enc']])
        samples_weights.append(row['weight'])

    full_train_data_arr = np.array(train_data)
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(
        train_data, batch_size=500, num_workers=1, sampler=sampler)

    for epoch in range(200):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data)
            out_loss = loss(out, np.argmax(target, axis=1))
            out_loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {out_loss.data.item():.6f}')

        with torch.no_grad():
            all_data = torch.Tensor(np.vstack(full_train_data_arr[:,0]))
            all_labels =  np.vstack(full_train_data_arr[:,1])
            out = model(all_data)
            y_pred = np.argmax(out, axis=1)
            y_true = np.argmax(all_labels, axis=1)
            print(f1_score(y_true, y_pred, average='macro'))


    torch.save(model, path_to_model)


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
    "-o",
    "--output",
    "output",
    type=click.Path(file_okay=True, exists=False),
    required=True,
)
@click.option(
    "-ob",
    "--output-binarizer",
    "output_binarizer",
    type=click.Path(file_okay=True, exists=False),
    required=True,
)
def train_models(train_polemo_embedded: str, val_polemo_embedded: str, test_polemo_embedded: str,
                 political_tweets_embedded: str, wordnet_embedded: str, output: str, output_binarizer: str):
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
    with open(output_binarizer, "wb") as model_file:
        pkl.dump(encoder, model_file, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train_models()  # pylint: disable=no-value-for-parameter
