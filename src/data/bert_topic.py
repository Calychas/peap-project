# from transformers import AutoTokenizer, AutoModel
import pandas as pd
import umap
import src.data.utils as utils
from sklearn.cluster import DBSCAN
import numpy as np
import os
import pickle as pkl
import gzip

embeddings = utils.list_full_paths(os.path.join("datasets", "embeddings"))
embeddings = np.vstack(list(map(lambda x: np.array(pd.read_pickle(x)["tweet_embedding"].to_list()), embeddings)))

umap_ble = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', low_memory=True)
umap_embeddings = umap_ble.fit_transform(embeddings)

with open(os.path.join("datasets", "account_tweets_umap_low.pkl"), 'wb') as f:
    pkl.dump(umap_ble, f)

with open(os.path.join("datasets", "account_tweets_umap_embeddings_low.pkl"), 'wb') as f:
    pkl.dump(umap_embeddings, f)

# with gzip.open(os.path.join("datasets", "umap_embeddings.pkl.gz"), 'rb') as f:
#     umap_embeddings = np.load(f)

# print(umap_embeddings)