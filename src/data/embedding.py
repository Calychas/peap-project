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
@click.option("-m", "--model", type=click.Choice(["herbert"]), required=True)
@click.option("-a", "--aggregation", type=click.Choice(["mean"]), required=True)
@click.option(
    "-i",
    "--input-dir",
    "input_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "-o", "--output-file", "output_file", type=click.Path(dir_okay=False), required=True
)
@click.option(
    "-p", "--partial-output-dir", "partial_output_dir", type=click.Path(file_okay=False, exists=False), required=False
)
def embed_accounts(model: str, aggregation: str, input_dir: str, output_file: str, partial_output_dir: str):
    save_partials = partial_output_dir is not None

    if save_partials:
        os.mkdir(partial_output_dir)

    if model == "herbert":
        tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
        model = AutoModel.from_pretrained("allegro/herbert-base-cased")
        if CUDA:
            model = model.to("cuda")

        if aggregation == "mean":
            agg_func = lambda embeddings: np.mean(embeddings, axis=0)

            csv_files = [
                f
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".csv")
            ]
            results = []
            i = 0
            for account_tweets_path in tqdm(csv_files):
                full_path = os.path.join(input_dir, account_tweets_path)
                user_tweets = pd.read_csv(full_path)
                embedding_results = embed_tweets(user_tweets, tokenizer, model, agg_func, save_partials)
                if save_partials:
                    username = embedding_results[0]
                    all_tweets_embeddings = embedding_results[-1]
                    with open(os.path.join(partial_output_dir, f"{username}.pkl"), 'wb') as f:
                        pkl.dump(all_tweets_embeddings, f)

                    user_embedding = embedding_results[:-1]
                else:
                    user_embedding = embedding_results
                results.append(user_embedding)

            result_df = pd.DataFrame(results, columns=["username", "embedding"])
            result_df.to_csv(output_file, index=False)

        else:
            raise NotImplementedError(
                f"Aggregation {aggregation} not implemented for model {model}."
            )
    else:
        raise NotImplementedError(f"Model {model} not implemented.")


def embed_tweets(
    tweets_data: pd.DataFrame, tokenizer, model, aggregation, return_partials=False
) -> Tuple[str, np.ndarray]:
    username = tweets_data.at[0, 'username']
    tweet_texts = tweets_data["tweet"].tolist()
    tweet_embeddings = []
    with torch.no_grad():
        for batch in tqdm(chunks(tweet_texts, 150)):
            tokenized_text = tokenizer.batch_encode_plus(
                batch, padding="longest", add_special_tokens=True, return_tensors="pt"
            )

            if CUDA:
                tokenized_text = tokenized_text.to("cuda")
            outputs = model(**tokenized_text)
            batch_embeddings = outputs[1].cpu().numpy()
            tweet_embeddings.extend(batch_embeddings)
    all_embeddings = np.vstack(tweet_embeddings)
    print(all_embeddings.shape)
    account_embedding = aggregation(all_embeddings)
    if return_partials:
        return username, account_embedding, all_embeddings
    else:
        return username, account_embedding


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    embed_accounts()
