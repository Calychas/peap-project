from transformers import AutoTokenizer, AutoModel
import click
import pandas as pd
import os
import numpy as np
from typing import Tuple
import torch
from tqdm import tqdm
import pickle as pkl

CUDA = torch.cuda.is_available()
print(CUDA)


@click.command("embed")
@click.option("-m", "--model", type=click.Choice(["herbert", "politicalHerBERT"]), required=True)
@click.option("-a", "--aggregation", type=click.Choice(["mean"]), required=True)
@click.option(
    "-i",
    "--input-tweets-pickle",
    "input_tweets_pickle",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "-o", "--output-file", "output_file", type=click.Path(dir_okay=False), required=True
)
@click.option(
    "-p", "--partial-output-dir", "partial_output_dir", type=click.Path(file_okay=False, exists=False), required=False
)
def embed_accounts(model: str, aggregation: str, input_tweets_pickle: str, output_file: str, partial_output_dir: str):
    save_partials = partial_output_dir is not None

    tweets_df = pd.read_pickle(input_tweets_pickle)

    if save_partials and not os.path.exists(partial_output_dir):
        os.mkdir(partial_output_dir)

    if model == "herbert" or model == "politicalHerBERT":
        tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
        if model == "herbert":
            model = AutoModel.from_pretrained("allegro/herbert-base-cased")
        elif model == "politicalHerBERT":
            model = AutoModel.from_pretrained(os.path.join("..", "trained_models", "politicalHerBERT"))
        if CUDA:
            model = model.to("cuda")

        if aggregation == "mean":
            agg_func = lambda embeddings: np.mean(embeddings, axis=0)

            users = tweets_df['username'].unique()
            results = []
            for user in tqdm(users):
                if not os.path.exists(os.path.join(partial_output_dir, f"{user}.pkl.gz")):
                    user_tweets = tweets_df[tweets_df['username'] == user]
                    embedding_results = embed_tweets(user_tweets, tokenizer, model, agg_func, save_partials)
                    if save_partials:
                        data_df = embedding_results[-1]
                        data_df.to_pickle(os.path.join(partial_output_dir, f"{user}.pkl.gz"))

                        account_embedding = embedding_results[0]
                    else:
                        account_embedding = embedding_results
                    results.append((user, account_embedding))
                else:
                    data_df = pd.read_pickle(os.path.join(partial_output_dir, f"{user}.pkl.gz"))
                    tweet_embeddings = list(data_df["tweet_embedding"].values)
                    all_embeddings = np.vstack(tweet_embeddings)
                    account_embedding = agg_func(all_embeddings)
                    results.append((user, account_embedding))

            result_df = pd.DataFrame(results, columns=["username", "embedding"])
            result_df.to_csv(output_file, index=False)

        else:
            raise NotImplementedError(
                f"Aggregation {aggregation} not implemented for model {model}."
            )
    else:
        raise NotImplementedError(f"Model {model} not implemented.")


def embed_tweets(tweets_data: pd.DataFrame, tokenizer, model, aggregation, return_partials=False
                 ) -> Tuple[np.ndarray]:
    tweet_indices = tweets_data.index.values.tolist()
    tweet_ids = tweets_data["id"].tolist()
    tweet_usernames = tweets_data["username"].tolist()
    tweet_texts = tweets_data["tweet"].tolist()
    tweet_embeddings = []
    with torch.no_grad():
        for batch in tqdm(chunks(tweet_texts, 150), total=len(tweet_texts) // 150 + 1):
            tokenized_text = tokenizer.batch_encode_plus(
                batch, padding="longest", add_special_tokens=True, return_tensors="pt"
            )

            if CUDA:
                tokenized_text = tokenized_text.to("cuda")
            outputs = model(**tokenized_text)
            batch_embeddings = outputs[1].cpu().numpy()
            tweet_embeddings.extend(batch_embeddings)

    data_df = pd.DataFrame(index=tweet_indices, data={"tweet_id": tweet_ids,
                                                      "username": tweet_usernames,
                                                      "tweet_embedding": tweet_embeddings})
    all_embeddings = np.vstack(tweet_embeddings)
    account_embedding = aggregation(all_embeddings)
    if return_partials:
        return account_embedding, data_df
    else:
        return account_embedding


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


if __name__ == "__main__":
    embed_accounts()
