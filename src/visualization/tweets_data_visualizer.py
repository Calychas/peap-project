import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys


def plot_tweets_counts(accounts: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 12))
    sns.histplot(data=accounts, x="tweets_count", binwidth=500)
    plt.title("Twitter activity (tweets, retweets, replies) for all accounts")
    plt.xlabel("Number of tweets for a user")
    plt.ylabel("Number of accounts")
    plt.savefig("reports/plots/all_replies_count_hist_full.png")
    plt.close()

    accounts_with_less_than_500_tweets = accounts[accounts['tweets_count'] <= 500]

    plt.figure(figsize=(12, 12))
    sns.histplot(data=accounts_with_less_than_500_tweets, x="tweets_count", binwidth=20)
    plt.title("Twitter activity (tweets, retweets, replies) for accounts with less than 500 posts")
    plt.xlabel("Number of tweets for a user")
    plt.ylabel("Number of accounts")
    plt.savefig("reports/plots/all_replies_count_less_than_500_hist.png")
    plt.close()


if __name__ == '__main__':
    accounts_processed_file_path = sys.argv[1]
    accounts_df = pd.read_csv(accounts_processed_file_path)
    plot_tweets_counts(accounts_df)
