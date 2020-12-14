import click
import pandas as pd
import os
import fasttext
import json
from typing.io import IO

from src.data.utils import emoji2text_tweet


def remove_quotes_from_saved_file(txt_path: str):
    text = ""
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line[0] == "\"" and line[-2] == "\"":
                line = line[1:]
                line = line[:-2] + "\n"
            text += line

    os.remove(txt_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.close()


def get_training_df_for_fasttext(train_polemo_path: str, political_tweets_path: str,
                                              emoji_mapping_file: IO):
    emoji_mapping = json.load(emoji_mapping_file)
    emoji_mapping_items = emoji_mapping.items()

    full_df = pd.DataFrame(columns=["label", "text"])
    for file_path in [train_polemo_path, political_tweets_path]:
        df = pd.read_csv(file_path)
        df = df[['label', 'text']]
        df['label'] = "__label__" + df['label']
        df['text'] = df['text'].apply(lambda text: emoji2text_tweet(text, emoji_mapping_items))
        df['text'] = df['text'].apply(lambda string: string.lower())
        df['text'] = df['text'].apply(lambda string: string.replace("#", ""))
        full_df = full_df.append(df)

    return full_df


def preprocess_polemo_files(train_file: str, val_file: str, test_file: str):
    replace_dict = {
        "__label__z_amb": "ambiguous",
        "__label__z_minus_m": "negative",
        "__label__z_plus_m": "positive",
        "__label__z_zero": "neutral"
    }

    output_files = {}
    for dataset_name, file_to_process in {"train": train_file, "val": val_file, "test": test_file}.items():
        lines = []
        for line in file_to_process:
            lines.append(line)

        labels = []
        texts = []
        for line in lines:
            labels.append(line[line.index("__label__"):-1])
            texts.append(line[:line.index("__label__")])

        polemo_data = pd.DataFrame(data={"text": texts, "label": labels})

        polemo_data['label'] = polemo_data['label'].replace(replace_dict)
        path = ".".join(file_to_process.name.split(".")[:-1]) + "_processed.csv"
        output_files[dataset_name] = path
        polemo_data.to_csv(path, index=False)

    return output_files


@click.command()
@click.option(
    "-tr",
    "--train-file-polemo",
    "train_file_polemo",
    type=click.File(mode="r", encoding="UTF-8"),
    required=True
)
@click.option(
    "-v",
    "--val-file-polemo",
    "val_file_polemo",
    type=click.File(mode="r", encoding="UTF-8"),
    required=True
)
@click.option(
    "-te",
    "--test-file-polemo",
    "test_file_polemo",
    type=click.File(mode="r", encoding="UTF-8"),
    required=True
)
@click.option(
    "-p",
    "--political_tweets_file",
    "political_tweets_file",
    type=click.File(mode="r", encoding="UTF-8"),
    required=True
)
@click.option(
    "-e",
    "--emoji-mapping",
    "emoji_mapping_file",
    type=click.File(mode="r", encoding="UTF-8"),
    required=True,
)
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, writable=True),
    required=True
)
def train_sentiment_model(train_file_polemo: str, val_file_polemo: str, test_file_polemo: str,
                          political_tweets_file: str, emoji_mapping_file: IO, output_file: str):
    processed_polemo_files_paths = preprocess_polemo_files(train_file_polemo, val_file_polemo, test_file_polemo)

    full_training_data = get_create_txt_training_file_for_fasttext(processed_polemo_files_paths['train'],
                                                                   political_tweets_file, emoji_mapping_file)
    full_training_data['row'] = full_training_data['label'] + " " + full_training_data['text']

    path_to_dir = political_tweets_file.name.split(os.path.sep)[:-1]
    path_to_dir.append("full_train_data.txt")

    path_for_full_training_data = str(os.path.sep).join(path_to_dir)
    full_training_data['row'].to_csv(path_for_full_training_data, index=False,
                                     header=False)
    remove_quotes_from_saved_file(path_for_full_training_data)

    model = fasttext.train_supervised(input=path_for_full_training_data,
                                      wordNgrams=5, neg=5, dim=300, lr=0.005, epoch=500, loss="ns", verbose=1,
                                      label_prefix='__label__', thread=2)
    model.save_model(output_file)


if __name__ == '__main__':
    train_sentiment_model()  # pylint: disable=no-value-for-parameter
