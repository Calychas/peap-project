import json

import click
import pandas as pd
import os
from tqdm import tqdm
import torch
import numpy as np
from typing.io import IO

from src.data.utils import emoji2text_tweet
from src.models.train_fasttext_model import preprocess_polemo_files
from src.models.fasttext_research import get_train_val_test_dataframes
from src.data.embedding import chunks

from transformers import AutoTokenizer, AutoModel

CUDA = torch.cuda.is_available()
print(CUDA)

def get_embeddings_for_df(df: pd.DataFrame, tokenizer, model):
    embeddings = []
    texts = list(df['text'])
    with torch.no_grad():
        for batch in tqdm(chunks(texts, 150), total=len(texts) // 150 + 1):
            tokenized_text = tokenizer.batch_encode_plus(
                batch, padding="longest", add_special_tokens=True, return_tensors="pt")
            if CUDA:
                tokenized_text = tokenized_text.to("cuda")
            outputs = model(**tokenized_text)
            batch_embeddings = outputs[1].cpu().detach().numpy()
            embeddings.extend(batch_embeddings)
    return np.vstack(embeddings)


@click.command()
@click.option(
    "-p",
    "--path-to-political-herbert",
    "path_to_political_herbert",
    type=click.Path(dir_okay=True, exists=True, readable=True),
    required=True,
)
@click.option(
    "-pc",
    "--path-to-polemo-conll",
    "path_to_polemo_conll",
    type=click.Path(dir_okay=True, exists=True, readable=True),
    required=True,
)
@click.option(
    "-t",
    "--tweet-annotations-path",
    "tweet_annotations_path",
    type=click.Path(dir_okay=False, exists=True, readable=True),
    required=True,
)
@click.option(
    "-w",
    "--wordnet-sentiment-path",
    "wordnet_sentiment_path",
    type=click.Path(dir_okay=False, exists=True, readable=True),
    required=True,
)
@click.option(
    "-e",
    "--emojis-file",
    "emojis_file",
    type=click.File(mode='r', encoding="utf-8"),
    required=True,
)
@click.option(
    "-s",
    "--sentiment-data-folder",
    "sentiment_data_folder",
    type=click.Path(dir_okay=True, exists=True, readable=True),
    required=True,
)
def process_sentiment_data_to_herbert(path_to_political_herbert: str, path_to_polemo_conll: str, tweet_annotations_path: str,
                                      wordnet_sentiment_path: str, emojis_file: IO, sentiment_data_folder: str):
    tokenizer_name = "allegro/herbert-klej-cased-tokenizer-v1"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(path_to_political_herbert, return_dict=True)
    if CUDA:
        model = model.to("cuda")

    train_file_polemo = open(os.path.join(path_to_polemo_conll, "all.sentence.train.txt"), 'r', encoding='utf-8')
    val_file_polemo = open(os.path.join(path_to_polemo_conll, "all.sentence.dev.txt"), 'r', encoding='utf-8')
    test_file_polemo = open(os.path.join(path_to_polemo_conll, "all.sentence.test.txt"), 'r', encoding='utf-8')

    processed_polemo_files_paths = preprocess_polemo_files(train_file_polemo, val_file_polemo, test_file_polemo)

    train_polemo_df = pd.read_csv(processed_polemo_files_paths['train'])
    val_polemo_df = pd.read_csv(processed_polemo_files_paths['val'])
    test_polemo_df = pd.read_csv(processed_polemo_files_paths['test'])
    political_tweets_df = pd.read_csv(tweet_annotations_path)
    wordnet_data_df = pd.read_csv(wordnet_sentiment_path)

    dfs = [train_polemo_df, val_polemo_df, test_polemo_df, political_tweets_df, wordnet_data_df]

    for df in dfs:
        print(df.columns)
        print(df['label'].unique())
        print(df['label'].value_counts())

    emoji_mapping = json.load(emojis_file)
    emoji_mapping_items = emoji_mapping.items()

    for df in dfs:
        df['text'] = df['text'].apply(lambda text: emoji2text_tweet(str(text), emoji_mapping_items))
        df['text'] = df['text'].apply(lambda text: text.lower())
        df['text'] = df['text'].apply(lambda text: text.replace("#", ""))

    train_polemo_df = dfs[0]
    val_polemo_df = dfs[1]
    test_polemo_df = dfs[2]
    political_tweets_df = dfs[3]
    wordnet_data_df = dfs[4]

    for name, df in {"train_polemo": train_polemo_df, "val_polemo": val_polemo_df, "test_polemo": test_polemo_df,
               "tweets": political_tweets_df, "wordnet": wordnet_data_df}.items():
        df_embeddings = get_embeddings_for_df(df, tokenizer, model)
        full_df = pd.DataFrame(data={
            "text" : list(df['text']),
            "label" : list(df['label']),
            "embeddings": list(df_embeddings)
        })
        full_df.to_pickle(f"{sentiment_data_folder}/{name}_herbert.pkl.gz")

if __name__ == "__main__":

    process_sentiment_data_to_herbert()  # pylint: disable=no-value-for-parameter
