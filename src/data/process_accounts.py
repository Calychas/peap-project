import os
import sys

import pandas as pd
from tqdm.auto import tqdm
from twitter_scraper import Profile


def read_accounts_and_clean(path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1])
    df = df.iloc[:, :5]
    df = df.dropna()

    df['username'] = df['link do konta'].apply(
        lambda x: x.replace('https://twitter.com/', '')
    )

    df['tweets_count'] = None

    return df


def get_tweets_count(accounts: pd.DataFrame):
    for _, account in tqdm(accounts.iterrows(), total=len(accounts)):
        if account['tweets_count'] is None:
            profile = Profile(account['username'])
            account['tweets_count'] = profile.tweets_count


if __name__ == '__main__':
    accounts_file_path = sys.argv[1]

    accounts = read_accounts_and_clean(accounts_file_path)

    while True:
        get_tweets_count(accounts)
        accounts.to_csv('datasets/accounts_processed.csv')

        if accounts['tweets_count'].isna().sum() == 0:
            break
