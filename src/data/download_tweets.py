import os
import sys

import pandas as pd
import twint
from tqdm import tqdm

TWEETS_LOWER_LIMIT = 20

def get_tweets_by_username(user: str):
    c = twint.Config()
    c.Username = user
    c.Output = f'datasets/tweets/{user}.csv'
    c.Store_csv = True
    c.Hide_output = True
    c.Limit = None
    c.Count = True

    twint.run.Search(c)


if __name__ == '__main__':
    accounts_info_path = sys.argv[1]

    os.mkdir('datasets/tweets')

    accounts = pd.read_csv(accounts_info_path)

    for _, account in tqdm(accounts.iterrows(), total=len(accounts)):
        if not os.path.isfile(f"datasets/tweets/{account['username']}.csv"):
            if account['tweets_count'] > TWEETS_LOWER_LIMIT:
                get_tweets_by_username(account['username'])
