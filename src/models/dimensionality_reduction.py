from operator import index
from turtle import mode
from umap import UMAP
from sklearn.manifold import TSNE
import numpy as np
from typing import Tuple
import pandas as pd
import os
import pickle as pkl

def create_umap_data_and_model(
    users_df: pd.DataFrame,
    output_path: str,
    data_file_name: str,
    model_file_name: str,
):
    embedding_column = "embedding"
    data = np.array(users_df[embedding_column].tolist())
    umap_2d, data_reduced_2d = perform_umap(data, 2)
    umap_3d, data_reduced_3d = perform_umap(data, 3)
    result_df = users_df.drop(columns=[embedding_column])
    result_df[['3D_x', '3D_y', '3D_z']] = data_reduced_3d.tolist()
    result_df[['2D_x', '2D_y']] = data_reduced_2d.tolist()

    data_file_output_path = os.path.join(output_path, f"{data_file_name}.csv")
    result_df.to_csv(data_file_output_path, index=False)

    def save_model(model: UMAP):
        model_output_path = os.path.join(output_path, f"{model_file_name}_{model.n_components}d.pkl.gz")
        with open(model_output_path, "wb") as f:
            pkl.dump(model, f)

    save_model(umap_2d)
    save_model(umap_3d)

def perform_umap(data: np.ndarray, n_dimensions: int) -> Tuple[UMAP, np.ndarray]:
    umap = UMAP(n_components=n_dimensions)
    reduced = umap.fit_transform(data)
    return umap, reduced


def perform_tsne(data: np.ndarray, n_dimensions: int) -> Tuple[TSNE, np.ndarray]:
    tsne = TSNE(n_components=n_dimensions)
    reduced = tsne.fit_transform(data)
    return tsne, reduced
