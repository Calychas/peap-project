import pickle as pkl

import click
import fasttext
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer


@click.command()
def prepare_data(
        tweets_input_file: str,
        users_input_file: str,
        sentiment_model_file: str,
        sentiment_tweets_file: str,
        topics_model_file: str,
        topics_vectorizer_file: str,
        topics_tweets_file: str,
        tweets_output_file: str,
        words_per_topic_file: str,
        topics_distribution_file: str,
        words_counts_file: str,
        sentiment_distributions_file: str
):
    click.echo("Preparing tweets file")

    tweets = pd.read_pickle(tweets_input_file)
    tweets = tweets[['username', 'id', 'link', 'tweet']]

    users = pd.read_csv(users_input_file, index_col=0)
    users = users[['username', 'party', 'coalition', 'pozycja']]
    users = users.rename(columns={'pozycja': 'role'})
    users['username'] = users['username'].apply(str.lower)

    tweets_users = tweets.merge(users, on='username')  # used

    del tweets
    del users

    click.echo("Merged with usernames/parties/coalitions")

    sentiment_model = fasttext.load_model(sentiment_model_file)

    tweets_for_sentiment = pd.read_pickle(sentiment_tweets_file)
    tweets_for_sentiment['tweet'] = tweets_for_sentiment['tweet'].apply(
        str.lower
    )
    tweets_for_sentiment = tweets_for_sentiment[['id', 'tweet']]

    just_tweets = tweets_for_sentiment['tweet'].tolist()
    predictions = sentiment_model.predict(just_tweets)[0]
    predictions = [label for sublist in predictions for label in sublist]

    tweets_for_sentiment['sentiment'] = predictions
    tweets_for_sentiment = tweets_for_sentiment[['id', 'sentiment']]

    tweets_users_sentiment = tweets_users.merge(  # used
        tweets_for_sentiment,
        on='id',
        how='right'
    )

    tweets_users_sentiment.replace(to_replace={
        '__label__positive': 'positive',
        '__label__negative': 'negative',
        '__label__ambiguous': 'ambiguous',
        '__label__neutral': 'neutral'
    }, inplace=True)

    del tweets_for_sentiment
    del tweets_users

    click.echo("Added sentiment for tweets")

    tweets_for_topics = pd.read_pickle(topics_tweets_file)

    with open(topics_vectorizer_file, 'rb') as vec_file:
        vectorizer: CountVectorizer = pkl.load(vec_file)

    with open(topics_model_file, 'rb') as lda_file:
        lda: LatentDirichletAllocation = pkl.load(lda_file)

    tweets_texts = tweets_for_topics.tweet.tolist()
    counts = vectorizer.transform(tweets_texts)

    probas = lda.transform(counts)  # used

    labels = np.argmax(probas, axis=1)
    prob_values = np.max(probas, axis=1)

    tweets_for_topics['topic'] = labels
    tweets_for_topics['topic_proba'] = prob_values

    tweets_for_topics = tweets_for_topics[['id', 'topic', 'topic_proba']]

    tweets_users_sentiment_topic = tweets_users_sentiment.merge(  # used
        tweets_for_topics,
        on='id'
    )

    del tweets_for_topics
    del tweets_users_sentiment

    click.echo("Added topics for tweets")

    tweets_users_sentiment_topic.to_pickle(tweets_output_file)

    click.echo(f"Stored processed tweets at {tweets_output_file}")

    words_in_topics = {}

    for topic_num, topic in enumerate(lda.components_):
        frequencies = [
            {
                'text': name,
                'value': freq
            }
            for name, freq in zip(vectorizer.get_feature_names(), topic)
        ]
        words_in_topics[topic_num] = frequencies

    with open(words_per_topic_file, 'wb') as f:
        pkl.dump(words_in_topics, f)

    click.echo("Calculating topics per username/party/coalition")

    clean_tweets = pd.read_pickle(topics_tweets_file)

    topics_count = len(lda.components_)

    a = clean_tweets.merge(tweets_users_sentiment_topic, on='id')
    a.rename(columns={'username_x': 'username'}, inplace=True)
    a = a.reset_index()

    del clean_tweets

    def get_topic_distribution_for_column(column_value, column_name):
        indices = np.array(a[a[column_name] == column_value].index.tolist())
        topics = probas[indices]
        values = np.sum(topics, axis=0)
        distribution = values / np.sum(values)
        return distribution

    topics_distributions = {
        'per_user': {},
        'per_party': {},
        'per_coalition': {}
    }

    unique_usernames = a.username.unique()
    unique_parties = a.party.unique()
    unique_coalitions = a.coalition.unique()

    for username in tqdm(unique_usernames):
        topics_distributions['per_user'][username] = [
            {
                'topic': t,
                'part': p
            }
            for t, p
            in zip(range(topics_count), get_topic_distribution_for_column(
                column_name='username',
                column_value=username))
        ]

    for party in tqdm(unique_parties):
        topics_distributions['per_party'][party] = [
            {
                'topic': t,
                'part': p
            }
            for t, p
            in zip(range(topics_count), get_topic_distribution_for_column(
                column_name='party',
                column_value=party))
        ]

    for coalition in tqdm(unique_coalitions):
        topics_distributions['per_coalition'][coalition] = [
            {
                'topic': t,
                'part': p
            }
            for t, p
            in zip(range(topics_count), get_topic_distribution_for_column(
                column_name='coalition',
                column_value=coalition))
        ]

    with open(topics_distribution_file, 'wb') as f:
        pkl.dump(topics_distributions, f)

    click.echo(f"Stored topics distributions at {topics_distribution_file}")

    click.echo("Calculating words per username/party/coalition")

    clean_tweets = pd.read_pickle(topics_tweets_file)
    a = clean_tweets.merge(
        tweets_users_sentiment_topic,
        on='id',
        suffixes=('', '_y'))
    a.reset_index(inplace=True)

    del clean_tweets

    counts = vectorizer.transform(a.tweet.tolist())

    def get_word_counts_for_column(column_name, column_value):
        indices = np.array(a[a[column_name] == column_value].index.tolist())
        words = counts[indices]
        summed = np.sum(words, axis=0)
        return np.array(summed).squeeze()

    words_counts = {
        'per_user': {},
        'per_party': {},
        'per_coalition': {}
    }

    for username in tqdm(unique_usernames):
        words_counts['per_user'][username] = [
            {
                'text': name,
                'value': freq
            }
            for name, freq
            in zip(
                vectorizer.get_feature_names(),
                get_word_counts_for_column(
                    column_name='username',
                    column_value=username
                )
            )
        ]

    for party in tqdm(unique_parties):
        words_counts['per_party'][party] = [
            {
                'text': name,
                'value': freq
            }
            for name, freq
            in zip(
                vectorizer.get_feature_names(),
                get_word_counts_for_column(
                    column_name='party',
                    column_value=party
                )
            )
        ]

    for coalition in tqdm(unique_coalitions):
        words_counts['per_coalition'][coalition] = [
            {
                'text': name,
                'value': freq
            }
            for name, freq
            in zip(
                vectorizer.get_feature_names(),
                get_word_counts_for_column(
                    column_name='coalition',
                    column_value=coalition
                )
            )
        ]

    with open(words_counts_file, 'wb') as f:
        pkl.dump(words_counts, f)

    click.echo(f"Stores words counts at {words_counts_file}")

    click.echo("Calculating sentiment per username/party/coalition/topic")

    a = tweets_users_sentiment_topic

    sent_values = a.sentiment.unique()

    def get_sentiment_distribution_by_column(column_name, column_value):
        sent_counts = a[a[column_name] == column_value].sentiment.value_counts()
        tweets_count = sent_counts.sum()
        result = []
        for sent in sent_values:
            if sent in sent_counts.index:
                result.append((sent, sent_counts[sent] / tweets_count))
            else:
                result.append((sent, 0))

        return result

    sentiment_distributions = {
        'per_user': {},
        'per_party': {},
        'per_coalition': {},
        'per_topic': {}
    }

    unique_usernames = a.username.unique()
    unique_parties = a.party.unique()
    unique_coalitions = a.coalition.unique()
    unique_topics = a.topic.unique()

    for username in tqdm(unique_usernames):
        sentiment_distributions['per_user'][username] = get_sentiment_distribution_by_column(
            column_name='username',
            column_value=username
        )

    for topic in tqdm(unique_topics):
        sentiment_distributions['per_topic'][topic] = get_sentiment_distribution_by_column(
            column_name='topic',
            column_value=topic
        )

    for party in tqdm(unique_parties):
        sentiment_distributions['per_party'][party] = get_sentiment_distribution_by_column(
            column_name='party',
            column_value=party
        )

    for coalition in tqdm(unique_coalitions):
        sentiment_distributions['per_coalition'][coalition] = get_sentiment_distribution_by_column(
            column_name='coalition',
            column_value=coalition
        )

    with(open(sentiment_distributions_file, 'wb')) as f:
        pkl.dump(sentiment_distributions, f)

    click.echo(f"Stored sentiment distributions at {sentiment_distributions_file}")
# #%% md
#
# ## Coalitions and parties
#
# ### Extract info about each party and coalition for quicker access
#
# #%%
#
# accounts = pd.read_csv('../datasets/accounts_processed.csv')
#
# #%%
#
# parties = accounts.groupby('party').max()
#
# #%%
#
# parties.reset_index(inplace=True)
# parties = parties[['party', 'coalition']]
#
# #%%
#
# parties.to_csv('../datasets/for_presentation/parties.csv', index=False)
#
# #%% md
#
# ## Graph positions
#
# ### t-SNE
#
# #%%
#
# tweets = pd.read_pickle('../datasets/for_presentation/tweets_with_party_coalition_sentiment_topic.pkl.gz')
# usernames = tweets.username.unique()
#
# #%%
#
# embedding_data = pd.read_csv('../datasets/embeddings.csv')
# embedding_data['username'] = embedding_data['username'].str.lower()
#
# #%%
#
# embedding_data = embedding_data[embedding_data['username'].isin(usernames)]
#
# #%%
#
# embeddings = np.array([np.array([np.float(i) for i in x.replace("]", "").replace("[", "").split()]) for x in embedding_data['embedding'].tolist()])
# embeddings.shape
#
# #%%
#
# %%time
#
# tsne3d = TSNE(n_components=3).fit_transform(embeddings)
#
# #%%
#
# %%time
#
# tsne2d = TSNE(n_components=2).fit_transform(embeddings)
#
# #%%
#
# embeddings_normalized = Normalizer().fit_transform(embeddings)
# embeddings_standardized = StandardScaler().fit_transform(embeddings)
#
# tsne3d_standardized = TSNE(n_components=3).fit_transform(embeddings_standardized)
# tsne3d_normalized = TSNE(n_components=3).fit_transform(embeddings_normalized)
#
# tsne2d_standardized = TSNE(n_components=2).fit_transform(embeddings_standardized)
# tsne2d_normalized = TSNE(n_components=2).fit_transform(embeddings_normalized)
#
# #%%
#
# graph_positions = pd.DataFrame(tsne3d, columns=['3D_x', '3D_y', '3D_z'])
#
# #%%
#
# graph_positions['2D_x'] = tsne2d[:, 0]
# graph_positions['2D_y'] = tsne2d[:, 1]
# graph_positions['username'] = usernames
#
# #%%
#
# graph_positions.to_csv('../datasets/for_presentation/graph_tsne.csv', index=False)
#
# #%% md
#
# ## Clusters
#
# ### KMeans
#
# #%%
#
# tweets = pd.read_pickle('../datasets/for_presentation/tweets_with_party_coalition_sentiment_topic.pkl.gz')
# usernames = tweets.username.unique()
#
# embedding_data = pd.read_csv('../datasets/embeddings.csv')
# embedding_data['username'] = embedding_data['username'].str.lower()
#
# embedding_data = embedding_data[embedding_data['username'].isin(usernames)]
#
# embeddings = np.array([np.array([np.float(i) for i in x.replace("]", "").replace("[", "").split()]) for x in embedding_data['embedding'].tolist()])
# embeddings.shape
#
# #%%
#
# clusters = KMeans(n_clusters=6).fit(embeddings)
#
# #%%
#
# df = pd.DataFrame(usernames, columns=['username'])
# df['kmeans_cluster'] = clusters.labels_
#
# #%%
#
# df.to_csv('../datasets/for_presentation/clusters.csv', index=False)
