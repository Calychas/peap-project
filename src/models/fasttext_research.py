from typing import List, Tuple

import pandas as pd
import numpy as np
import os
import fasttext
import json
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.data.utils import emoji2text_tweet

np.random.seed(42)
PATH_TO_DATASETS = os.path.join("..", "..", "datasets")
PATH_TO_POLEMO_CONLL = os.path.join("..", "..", "datasets", "polemo", "dataset_conll")
PATH_TO_RESULTS = os.path.join("..", "..", "reports", "sentiment_analysis")
pretrained_vectors_path = os.path.join("..", "..", "trained_models", "kgr10.plain.skipgram.dim300.neg10.vec")

with open(os.path.join("..", "..", "datasets", "emojis.json"), encoding="utf-8") as f:
    emoji_mapping = json.load(f)

emoji_mapping_items = emoji_mapping.items()


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


def process_dataframe_to_fasttext_format(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe[['label', 'text']]
    dataframe['label'] = "__label__" + dataframe['label']
    dataframe = dataframe.astype(str)
    dataframe['text'] = dataframe['text'].apply(lambda string: str(emoji2text_tweet(str(string), emoji_mapping_items)))
    dataframe['text'] = dataframe['text'].apply(lambda string: str(string).lower())
    dataframe['text'] = dataframe['text'].apply(lambda string: str(string).replace("#", ""))
    dataframe['row'] = dataframe['label'] + " " + dataframe['text']
    return dataframe


def get_train_val_test_dataframes(texts: List, labels: List, train_size: float, val_size: float, test_size: float) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert train_size + val_size + test_size == 1.0

    texts_train_val, texts_test, labels_train_val, labels_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42)

    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_train_val,
        labels_train_val,
        test_size=val_size / (1 - test_size),
        random_state=42)

    train_df = pd.DataFrame(data={"text": texts_train, "label": labels_train})
    val_df = pd.DataFrame(data={"text": texts_val, "label": labels_val})
    test_df = pd.DataFrame(data={"text": texts_test, "label": labels_test})
    return train_df, val_df, test_df


def get_dataframes_for_all_files(files_dict: dict) -> dict:
    data_for_fasttext = {}
    for dataset, file_path in files_dict.items():
        df = pd.read_csv(file_path)
        print(f"Processing {dataset} dataset")
        df = process_dataframe_to_fasttext_format(df)
        path = os.path.join(PATH_TO_DATASETS, "sentiment_data", f"{dataset}_data.txt")
        df['row'].to_csv(path, index=False, header=False)

        remove_quotes_from_saved_file(path)
        data_for_fasttext[dataset] = {}
        data_for_fasttext[dataset]["labels"] = list(df['label'].values)
        data_for_fasttext[dataset]["texts"] = list(df['text'].values)
        data_for_fasttext[dataset]["dataframe"] = df

    return data_for_fasttext


def save_datasets_to_txt_files(train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame,
                               train_set_file_name: str, val_set_file_name: str, test_set_file_name: str):
    train_set['row'] = train_set['label'] + " " + train_set['text']
    val_set['row'] = val_set['label'] + " " + val_set['text']
    test_set['row'] = test_set['label'] + " " + test_set['text']

    train_set['row'].to_csv(os.path.join(PATH_TO_DATASETS, "sentiment_data", train_set_file_name),
                            index=False, header=False)
    val_set['row'].to_csv(os.path.join(PATH_TO_DATASETS, "sentiment_data", val_set_file_name),
                          index=False, header=False)
    test_set['row'].to_csv(os.path.join(PATH_TO_DATASETS, "sentiment_data", test_set_file_name),
                           index=False, header=False)

    remove_quotes_from_saved_file(os.path.join(PATH_TO_DATASETS, "sentiment_data", train_set_file_name))
    remove_quotes_from_saved_file(os.path.join(PATH_TO_DATASETS, "sentiment_data", val_set_file_name))
    remove_quotes_from_saved_file(os.path.join(PATH_TO_DATASETS, "sentiment_data", test_set_file_name))


def train_default_parameters_fasttext(val_polemo_texts: List, val_polemo_labels: List, val_wordnet_texts: List,
                                      val_wordnet_labels: List, val_tweets_texts: List, val_tweets_labels: List,
                                      file_name: str, use_pretrained_vector: bool, loss, neg=5, dim=300,
                                      how_many_runs=10, ngram=1, lr=0.1, epoch=5) -> Tuple[
    float, float, float]:
    polemo_res = []
    tweets_res = []
    wordnet_res = []
    for s in range(how_many_runs):
        seed = s * 10
        if use_pretrained_vector:
            model = fasttext.train_supervised(input=os.path.join(PATH_TO_DATASETS, "sentiment_data", file_name),
                                              loss=loss, neg=neg, wordNgrams=ngram,
                                              verbose=2, label_prefix='__label__',
                                              thread=1, seed=seed, dim=300, lr=lr, epoch=epoch,
                                              pretrainedVectors=pretrained_vectors_path)
        else:
            model = fasttext.train_supervised(input=os.path.join(PATH_TO_DATASETS, "sentiment_data", file_name),
                                              loss=loss, neg=neg, wordNgrams=ngram, lr=lr, epoch=epoch,
                                              verbose=2, label_prefix='__label__',
                                              thread=1, seed=seed, dim=dim)

        polemo_res.append(f1_score(model.predict(val_polemo_texts)[0], val_polemo_labels, average='macro'))
        wordnet_res.append(f1_score(model.predict(val_wordnet_texts)[0], val_wordnet_labels, average='macro'))
        tweets_res.append(f1_score(model.predict(val_tweets_texts)[0], val_tweets_labels, average='macro'))

    polemo_f1_score, tweets_f1_score, wordnet_f1_score = np.mean(polemo_res), np.mean(tweets_res), np.mean(wordnet_res)
    print(f"Training file: {file_name}")
    print(f"Dim={dim}")
    print(f"loss={loss}")
    print(f"neg={neg}")
    print(f"ngram={ngram}")
    print(f"lr={lr}")
    print(f"epoch={epoch}")
    print(f"Use pretrained vectors: {use_pretrained_vector}")
    print(
        f"Average F1-score for test set of PolEmo: {polemo_f1_score}")
    print(
        f"Average F1-score for test set of political tweets: {tweets_f1_score}")
    print(
        f"Average F1-score for test set of WordNet data: {wordnet_f1_score}")
    print()
    return polemo_f1_score, tweets_f1_score, wordnet_f1_score


def comparison_of_training_set_dim_and_pretrained_softmax(val_polemo_texts: List, val_polemo_labels: List,
                                                          val_wordnet_texts: List,
                                                          val_wordnet_labels: List, val_tweets_texts: List,
                                                          val_tweets_labels: List):
    dims = []
    use_pres = []
    train_files = []
    polemo_f1_scores = []
    tweets_f1_scores = []
    wordnet_f1_scores = []

    list_of_files = ["train_polemo_tweets_data.txt", "train_wordnet_tweets_data.txt", "full_train_data.txt"]

    for dim in [100, 300, 500, 1000]:
        for dataset_file in list_of_files:
            polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                                   val_polemo_labels,
                                                                                                   val_wordnet_texts,
                                                                                                   val_wordnet_labels,
                                                                                                   val_tweets_texts,
                                                                                                   val_tweets_labels,
                                                                                                   dataset_file, False,
                                                                                                   dim=dim,
                                                                                                   loss="softmax",
                                                                                                   neg=5)
            dims.append(dim)
            use_pres.append(False)
            train_files.append(dataset_file)
            polemo_f1_scores.append(polemo_f1_score)
            tweets_f1_scores.append(tweets_f1_score)
            wordnet_f1_scores.append(wordnet_f1_score)

    for dataset_file in list_of_files:
        polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                               val_polemo_labels,
                                                                                               val_wordnet_texts,
                                                                                               val_wordnet_labels,
                                                                                               val_tweets_texts,
                                                                                               val_tweets_labels,
                                                                                               dataset_file, True,
                                                                                               dim=300, loss="softmax",
                                                                                               neg=5)
        dims.append(300)
        use_pres.append(True)
        train_files.append(dataset_file)
        polemo_f1_scores.append(polemo_f1_score)
        tweets_f1_scores.append(tweets_f1_score)
        wordnet_f1_scores.append(wordnet_f1_score)

    results = pd.DataFrame(data={"dim": dims,
                                 "train_file": train_files,
                                 "use_pretrained_vector": use_pres,
                                 "val_tweets_f1_score": tweets_f1_scores,
                                 "val_polemo_f1_score": polemo_f1_scores,
                                 "val_wordnet_f1_score": wordnet_f1_scores})

    results.to_csv(os.path.join(PATH_TO_RESULTS, "comparison_of_training_sets_dimension_and_pretrained_softmax.csv"),
                   index=False)


def comparison_of_training_set_dim_and_pretrained_ns(val_polemo_texts: List, val_polemo_labels: List,
                                                     val_wordnet_texts: List,
                                                     val_wordnet_labels: List, val_tweets_texts: List,
                                                     val_tweets_labels: List):
    dims = []
    negs = []
    use_pres = []
    train_files = []
    polemo_f1_scores = []
    tweets_f1_scores = []
    wordnet_f1_scores = []

    list_of_files = ["train_polemo_tweets_data.txt", "train_wordnet_tweets_data.txt", "full_train_data.txt"]

    for neg in [5, 25, 50]:
        for dim in [100, 300, 500, 1000]:
            for dataset_file in list_of_files:
                polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                                       val_polemo_labels,
                                                                                                       val_wordnet_texts,
                                                                                                       val_wordnet_labels,
                                                                                                       val_tweets_texts,
                                                                                                       val_tweets_labels,
                                                                                                       dataset_file,
                                                                                                       False,
                                                                                                       dim=dim,
                                                                                                       loss="ns",
                                                                                                       neg=neg)
                dims.append(dim)
                use_pres.append(False)
                negs.append(neg)
                train_files.append(dataset_file)
                polemo_f1_scores.append(polemo_f1_score)
                tweets_f1_scores.append(tweets_f1_score)
                wordnet_f1_scores.append(wordnet_f1_score)

        for dataset_file in list_of_files:
            polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                                   val_polemo_labels,
                                                                                                   val_wordnet_texts,
                                                                                                   val_wordnet_labels,
                                                                                                   val_tweets_texts,
                                                                                                   val_tweets_labels,
                                                                                                   dataset_file, True,
                                                                                                   dim=300, loss="ns",
                                                                                                   neg=neg)
            dims.append(300)
            use_pres.append(True)
            negs.append(neg)
            train_files.append(dataset_file)
            polemo_f1_scores.append(polemo_f1_score)
            tweets_f1_scores.append(tweets_f1_score)
            wordnet_f1_scores.append(wordnet_f1_score)

    results = pd.DataFrame(data={"dim": dims,
                                 "neg": negs,
                                 "train_file": train_files,
                                 "use_pretrained_vector": use_pres,
                                 "val_tweets_f1_score": tweets_f1_scores,
                                 "val_polemo_f1_score": polemo_f1_scores,
                                 "val_wordnet_f1_score": wordnet_f1_scores})

    results.to_csv(os.path.join(PATH_TO_RESULTS, "comparison_of_training_sets_dimension_and_pretrained_ns.csv"),
                   index=False)


def ngrams_search_softmax(val_polemo_texts: List, val_polemo_labels: List,
                          val_wordnet_texts: List,
                          val_wordnet_labels: List, val_tweets_texts: List,
                          val_tweets_labels: List):
    ngrams = []
    polemo_f1_scores = []
    tweets_f1_scores = []
    wordnet_f1_scores = []
    for ngram in [1, 2, 3, 4, 5, 6, 7]:
        polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                               val_polemo_labels,
                                                                                               val_wordnet_texts,
                                                                                               val_wordnet_labels,
                                                                                               val_tweets_texts,
                                                                                               val_tweets_labels,
                                                                                               "train_wordnet_tweets_data.txt",
                                                                                               True,
                                                                                               dim=300, loss="softmax",
                                                                                               ngram=ngram,
                                                                                               how_many_runs=5)
        ngrams.append(ngram)
        polemo_f1_scores.append(polemo_f1_score)
        tweets_f1_scores.append(tweets_f1_score)
        wordnet_f1_scores.append(wordnet_f1_score)

    results = pd.DataFrame(data={"ngram": ngrams,
                                 "val_tweets_f1_score": tweets_f1_scores,
                                 "val_polemo_f1_score": polemo_f1_scores,
                                 "val_wordnet_f1_score": wordnet_f1_scores})

    results.to_csv(os.path.join(PATH_TO_RESULTS, "comparison_of_ngrams_softmax.csv"), index=False)


def ngrams_search_ns(val_polemo_texts: List, val_polemo_labels: List,
                     val_wordnet_texts: List,
                     val_wordnet_labels: List, val_tweets_texts: List,
                     val_tweets_labels: List):
    ngrams = []
    polemo_f1_scores = []
    tweets_f1_scores = []
    wordnet_f1_scores = []
    for ngram in [1, 2, 3, 4, 5, 6, 7]:
        polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                               val_polemo_labels,
                                                                                               val_wordnet_texts,
                                                                                               val_wordnet_labels,
                                                                                               val_tweets_texts,
                                                                                               val_tweets_labels,
                                                                                               "train_polemo_tweets_data.txt",
                                                                                               False,
                                                                                               dim=500, loss="ns",
                                                                                               ngram=ngram,
                                                                                               how_many_runs=5)
        ngrams.append(ngram)
        polemo_f1_scores.append(polemo_f1_score)
        tweets_f1_scores.append(tweets_f1_score)
        wordnet_f1_scores.append(wordnet_f1_score)

    results = pd.DataFrame(data={"ngram": ngrams,
                                 "val_tweets_f1_score": tweets_f1_scores,
                                 "val_polemo_f1_score": polemo_f1_scores,
                                 "val_wordnet_f1_score": wordnet_f1_scores})

    results.to_csv(os.path.join(PATH_TO_RESULTS, "comparison_of_ngrams_ns.csv"), index=False)



def lr_and_epochs_search_softmax(val_polemo_texts: List, val_polemo_labels: List,
                            val_wordnet_texts: List,
                            val_wordnet_labels: List, val_tweets_texts: List,
                            val_tweets_labels: List):
    lrs = []
    epochs = []
    polemo_f1_scores = []
    tweets_f1_scores = []
    wordnet_f1_scores = []
    for lr in [0.1, 0.2, 0.3]:
        for epoch in [2, 5, 7, 10]:
            polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                                   val_polemo_labels,
                                                                                                   val_wordnet_texts,
                                                                                                   val_wordnet_labels,
                                                                                                   val_tweets_texts,
                                                                                                   val_tweets_labels,
                                                                                                   "train_wordnet_tweets_data.txt",
                                                                                                   True,
                                                                                                   lr=lr,
                                                                                                   epoch=epoch,
                                                                                                   dim=300, loss="softmax",
                                                                                                   ngram=2,
                                                                                                   how_many_runs=1)
            lrs.append(lr)
            epochs.append(epoch)
            polemo_f1_scores.append(polemo_f1_score)
            tweets_f1_scores.append(tweets_f1_score)
            wordnet_f1_scores.append(wordnet_f1_score)

    for lr in [0.001, 0.002, 0.003]:
        for epoch in [200, 500, 700, 1000]:
            polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                                   val_polemo_labels,
                                                                                                   val_wordnet_texts,
                                                                                                   val_wordnet_labels,
                                                                                                   val_tweets_texts,
                                                                                                   val_tweets_labels,
                                                                                                   "train_wordnet_tweets_data.txt",
                                                                                                   True,
                                                                                                   lr=lr,
                                                                                                   epoch=epoch,
                                                                                                   dim=300, loss="softmax",
                                                                                                   ngram=2,
                                                                                                   how_many_runs=1)
            lrs.append(lr)
            epochs.append(epoch)
            polemo_f1_scores.append(polemo_f1_score)
            tweets_f1_scores.append(tweets_f1_score)
            wordnet_f1_scores.append(wordnet_f1_score)

    lr_epoch_results = pd.DataFrame(data={"lr": lrs,
                                          "epoch": epochs,
                                          "val_tweets_f1_score": tweets_f1_scores,
                                          "val_polemo_f1_score": polemo_f1_scores,
                                          "val_wordnet_f1_score": wordnet_f1_scores})

    lr_epoch_results.to_csv(os.path.join(PATH_TO_RESULTS, "comparison_of_lr_epoch_softmax.csv"),
                            index=False)


def lr_and_epochs_search_ns(val_polemo_texts: List, val_polemo_labels: List,
                            val_wordnet_texts: List,
                            val_wordnet_labels: List, val_tweets_texts: List,
                            val_tweets_labels: List):
    lrs = []
    epochs = []
    polemo_f1_scores = []
    tweets_f1_scores = []
    wordnet_f1_scores = []
    for lr in [0.1, 0.2, 0.3]:
        for epoch in [2, 5, 7, 10]:
            polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                                   val_polemo_labels,
                                                                                                   val_wordnet_texts,
                                                                                                   val_wordnet_labels,
                                                                                                   val_tweets_texts,
                                                                                                   val_tweets_labels,
                                                                                                   "train_polemo_tweets_data.txt",
                                                                                                   False,
                                                                                                   lr=lr,
                                                                                                   epoch=epoch,
                                                                                                   dim=500,
                                                                                                   loss="ns",
                                                                                                   ngram=1,
                                                                                                   how_many_runs=5)
            lrs.append(lr)
            epochs.append(epoch)
            polemo_f1_scores.append(polemo_f1_score)
            tweets_f1_scores.append(tweets_f1_score)
            wordnet_f1_scores.append(wordnet_f1_score)

    for lr in [0.001, 0.002, 0.003]:
        for epoch in [200, 500, 700, 1000]:
            polemo_f1_score, tweets_f1_score, wordnet_f1_score = train_default_parameters_fasttext(val_polemo_texts,
                                                                                                   val_polemo_labels,
                                                                                                   val_wordnet_texts,
                                                                                                   val_wordnet_labels,
                                                                                                   val_tweets_texts,
                                                                                                   val_tweets_labels,
                                                                                                   "train_polemo_tweets_data.txt",
                                                                                                   False,
                                                                                                   lr=lr,
                                                                                                   epoch=epoch,
                                                                                                   dim=500,
                                                                                                   loss="ns",
                                                                                                   ngram=1,
                                                                                                   how_many_runs=5)
            lrs.append(lr)
            epochs.append(epoch)
            polemo_f1_scores.append(polemo_f1_score)
            tweets_f1_scores.append(tweets_f1_score)
            wordnet_f1_scores.append(wordnet_f1_score)

    lr_epoch_results = pd.DataFrame(data={"lr": lrs,
                                          "epoch": epochs,
                                          "val_tweets_f1_score": tweets_f1_scores,
                                          "val_polemo_f1_score": polemo_f1_scores,
                                          "val_wordnet_f1_score": wordnet_f1_scores})

    lr_epoch_results.to_csv(os.path.join(PATH_TO_RESULTS, "comparison_of_lr_epoch_ns.csv"),
                            index=False)


if __name__ == '__main__':
    files = {
        "train_polemo": os.path.join(PATH_TO_POLEMO_CONLL, "all.sentence.train_processed.csv"),
        "dev_polemo": os.path.join(PATH_TO_POLEMO_CONLL, "all.sentence.dev_processed.csv"),
        "test_polemo": os.path.join(PATH_TO_POLEMO_CONLL, "all.sentence.test_processed.csv"),
        "annotation": os.path.join(PATH_TO_DATASETS, "sentiment_data", "political_tweets_annotations.csv"),
        "wordnet_sentiment": os.path.join(PATH_TO_DATASETS, "sentiment_data", "sentiment_from_plwordnet.csv")
    }

    dicts_dfs = get_dataframes_for_all_files(files)

    for dataset_name, dataset_data in dicts_dfs.items():
        print(f"Number of rows for {dataset_name}: {len(dataset_data['dataframe'])}")

    train_tweets, val_tweets, test_tweets = get_train_val_test_dataframes(dicts_dfs['annotation']['texts'],
                                                                          dicts_dfs['annotation']['labels'],
                                                                          train_size=0.8, val_size=0.1, test_size=0.1)

    train_wordnet, val_wordnet, test_wordnet = get_train_val_test_dataframes(dicts_dfs['wordnet_sentiment']['texts'],
                                                                             dicts_dfs['wordnet_sentiment']['labels'],
                                                                             train_size=0.8, val_size=0.1,
                                                                             test_size=0.1)

    train_polemo = dicts_dfs['train_polemo']['dataframe'][["text", "label"]]
    val_polemo = dicts_dfs['dev_polemo']['dataframe'][["text", "label"]]
    test_polemo = dicts_dfs['test_polemo']['dataframe'][["text", "label"]]

    # save_datasets_to_txt_files(train_wordnet, val_wordnet, test_wordnet, "train_wordnet.txt", "val_wordnet.txt",
    #                            "test_wordnet.txt")
    # save_datasets_to_txt_files(pd.concat([train_polemo, train_tweets]), pd.concat([val_polemo, val_tweets]),
    #                            pd.concat([test_polemo, test_tweets]),
    #                            "train_polemo_tweets_data.txt", "val_polemo_tweets_data.txt",
    #                            "test_polemo_tweets_data.txt")
    # save_datasets_to_txt_files(pd.concat([train_wordnet, train_tweets]), pd.concat([val_wordnet, val_tweets]),
    #                            pd.concat([test_wordnet, test_tweets]),
    #                            "train_wordnet_tweets_data.txt", "val_wordnet_tweets_data.txt",
    #                            "test_wordnet_tweets_data.txt")
    # save_datasets_to_txt_files(pd.concat([train_polemo, train_wordnet, train_tweets]),
    #                            pd.concat([val_polemo, val_wordnet, val_tweets]),
    #                            pd.concat([test_polemo, test_wordnet, test_tweets]),
    #                            "full_train_data.txt", "full_val_data.txt", "full_test_data.txt")

    # UNCOMMENT FUNCTIONS WHICH YOU WANT TO TEST
    #
    # comparison_of_training_set_dim_and_pretrained_softmax(list(val_polemo['text'].values), list(val_polemo['label'].values),
    #                                               list(val_wordnet['text'].values), list(val_wordnet['label'].values),
    #                                               list(val_tweets['text'].values), list(val_tweets['label'].values))
    # comparison_of_training_set_dim_and_pretrained_ns(list(val_polemo['text'].values), list(val_polemo['label'].values),
    #                                               list(val_wordnet['text'].values), list(val_wordnet['label'].values),
    #                                               list(val_tweets['text'].values), list(val_tweets['label'].values))
    # ngrams_search_softmax(list(val_polemo['text'].values), list(val_polemo['label'].values),
    #               list(val_wordnet['text'].values), list(val_wordnet['label'].values),
    #               list(val_tweets['text'].values), list(val_tweets['label'].values))
    # ngrams_search_ns(list(val_polemo['text'].values), list(val_polemo['label'].values),
    #                  list(val_wordnet['text'].values), list(val_wordnet['label'].values),
    #                  list(val_tweets['text'].values), list(val_tweets['label'].values))
    # lr_and_epochs_search_softmax(list(val_polemo['text'].values), list(val_polemo['label'].values),
    #                  list(val_wordnet['text'].values), list(val_wordnet['label'].values),
    #                  list(val_tweets['text'].values), list(val_tweets['label'].values))
    # lr_and_epochs_search_ns(list(val_polemo['text'].values), list(val_polemo['label'].values),
    #                  list(val_wordnet['text'].values), list(val_wordnet['label'].values),
    #                  list(val_tweets['text'].values), list(val_tweets['label'].values))
