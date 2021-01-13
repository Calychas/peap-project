# from transformers import AutoTokenizer, AutoModel
import pandas as pd
import umap
import src.data.utils as utils
from sklearn.cluster import DBSCAN
import numpy as np
import os
import gzip

embeddings = utils.list_full_paths(os.path.join("datasets", "embeddings"))
embeddings = np.vstack(list(map(lambda x: pd.read_pickle(x), embeddings)))

umap_embeddings = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', low_memory=True).fit_transform(embeddings)

with gzip.open(os.path.join("datasets", "umap_embeddings_low.pkl.gz"), 'wb') as f:
    print(umap_embeddings)
    np.save(f, umap_embeddings)

# with gzip.open(os.path.join("datasets", "umap_embeddings.pkl.gz"), 'rb') as f:
#     umap_embeddings = np.load(f)

# print(umap_embeddings)