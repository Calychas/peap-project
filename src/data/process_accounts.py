import sys

import pandas as pd
from tqdm import trange
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


def update_tweets_count_in_df(accounts_df: pd.DataFrame):
    pbar = trange(len(accounts_df))
    for _, account in accounts_df.iterrows():
        if account['tweets_count'] is None:
            profile = Profile(account['username'])
            account['tweets_count'] = profile.tweets_count
            if account['tweets_count'] is not None:
                pbar.update(1)

    pbar.close()


if __name__ == '__main__':
    accounts_file_path = sys.argv[1]

    accounts = read_accounts_and_clean(accounts_file_path)

    while accounts['tweets_count'].isna().sum() != 0:
        update_tweets_count_in_df(accounts)
        accounts.to_csv('datasets/accounts_processed.csv')
