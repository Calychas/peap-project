from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np



def cluster_k_means(data: np.ndarray, n_clusters: int):
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(data)
    return model, labels

def cluster_gmm(data: np.ndarray, n_clusters: int):
    model = GaussianMixture(n_components=n_clusters)
    labels = model.fit_predict(data)
    return model, labels


def cluster_db_scan(data: np.ndarray):
    pass #TODO: calculate distances in the embedding space to choose good epsilon