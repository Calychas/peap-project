import pandas as pd


def read_accounts_and_clean() -> pd.DataFrame:
    df = pd.read_csv('../../datasets/accounts.csv', skiprows=[1])
    df = df.iloc[:, :5]
    df = df.dropna()

    return df


if __name__ == '__main__':
    accounts = read_accounts_and_clean()

    print(accounts)
