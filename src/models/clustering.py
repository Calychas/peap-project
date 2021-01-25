from turtle import mode
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
from typing import Tuple, Dict, Union
import pandas as pd
import pickle as pkl
import os
from sklearn.preprocessing import Normalizer

def create_clustering_files(
    clustering_results: Tuple[
        pd.DataFrame, Dict[str, Union[KMeans, GaussianMixture, MeanShift]]
    ],
    output_path: str,
    labels_file_name: str,
    models_file_name: str,
):
    df, models = clustering_results
    labels_path = os.path.join(output_path, f"{labels_file_name}.csv")
    models_path = os.path.join(output_path, f"{models_file_name}.pkl.gz")
    df.to_csv(labels_path, index=False)
    with open(models_path, "wb") as f:
        pkl.dump(models, f)


def perform_clusterings(
    users_dataframe: pd.DataFrame, k_means_clusters: int, gmm_clusters: int, mean_shift_bandwith: float
) -> Tuple[pd.DataFrame, Dict[str, Union[KMeans, GaussianMixture, MeanShift]]]:
    embedding_column = "embedding"
    data = np.array(users_dataframe[embedding_column].tolist())

    k_means_model, k_means_labels = cluster_k_means(data, k_means_clusters)
    gmm_model, gmm_labels = cluster_gmm(data, gmm_clusters)
    mean_shift_model, mean_shift_labels = cluster_mean_shift(data, mean_shift_bandwith)

    models_dict = {"k_means": k_means_model, "gmm": gmm_model, "mean_shift": mean_shift_model}

    clusters_df = users_dataframe.drop(columns=[embedding_column])
    clusters_df["kmeans_cluster"] = k_means_labels
    clusters_df["gmm_cluster"] = gmm_labels
    clusters_df["mean_shift_cluster"] = mean_shift_labels

    return clusters_df, models_dict


def cluster_k_means(data: np.ndarray, n_clusters: int):
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(data)
    return model, labels


def cluster_gmm(data: np.ndarray, n_clusters: int):
    model = GaussianMixture(n_components=n_clusters)
    labels = model.fit_predict(data)
    return model, labels


def cluster_db_scan(data: np.ndarray, eps: float):
    model = (
        DBSCAN(eps=eps)
    )  # TODO: calculate distances in the embedding space to choose good epsilon
    labels = model.fit_predict(data)
    return model, labels

def cluster_mean_shift(data: np.ndarray, bandwith: float):
    model = MeanShift(bandwidth=bandwith)
    labels = model.fit_predict(data)
    return model, labels
