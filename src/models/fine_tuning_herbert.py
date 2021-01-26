import os

import click
import torch
import pandas as pd
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, \
    LineByLineTextDataset, Trainer, TrainingArguments


@click.command()
@click.option(
    "-i",
    "--input-file-tweets",
    "input_file_tweets",
    type=click.Path(exists=True, dir_okay=False),
    required=True
)
@click.option(
    "-o",
    "--output-directory",
    "output_directory",
    type=click.Path(dir_okay=True, writable=True, exists=False),
    required=True
)
@click.option(
    "-t",
    "--temp-tweets-file-path",
    "temp_tweets_file_path",
    type=click.Path(dir_okay=False, writable=True, exists=False),
    required=True
)
def fine_tune_herbert(input_file_tweets: str, output_directory: str, temp_tweets_file_path: str):
    tokenizer_name = "allegro/herbert-klej-cased-tokenizer-v1"
    embedder_model = "allegro/herbert-klej-cased-v1"

    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    model = RobertaForMaskedLM.from_pretrained(embedder_model, return_dict=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    df = pd.read_pickle(input_file_tweets)

    df['tweet'].to_csv(temp_tweets_file_path, index=False, header=False)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=temp_tweets_file_path,
        block_size=70,
    )

    tr_args = TrainingArguments(
        output_dir=output_directory,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=84,
        load_best_model_at_end=True,
        save_steps=1_000,
        save_total_limit=10
    )
    trainer = Trainer(
        model=model,
        args=tr_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()

    trainer.save_model(output_directory)

    os.remove(temp_tweets_file_path)


if __name__ == '__main__':
    fine_tune_herbert()  # pylint: disable=no-value-for-parameter
