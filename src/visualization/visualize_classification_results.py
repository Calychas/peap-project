import os
from typing import List

import click
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def visualize_results(model_name: str, model_name_to_file: str, path: str, labels: List[str], directory_for_plots: str,
                      columns_to_plot: List[str]):
    if not os.path.exists(path):
        click.echo(message=f"File with {model_name} results not found on path - {path}")
        return

    results_df = pd.read_csv(path, sep=",")
    results_df["feature_type"].replace({"scaled": "Standardized", "original": "Original", "minmax": "Normalized"}, inplace=True)
    results_df.rename(columns={'feature_type': 'Feature Type', 'val_f1_score': "Validation F1-score"}, inplace=True)

    for label in labels:
        results_filtered = results_df[results_df['label_name'] == label]
        for column in columns_to_plot:
            plt.figure(figsize=(10,7))
            sns.boxplot(data=results_filtered, y='Validation F1-score', x=column, hue='Feature Type')
            plt.title(f"Results for {model_name} model, output feature - {label}, column - {column}")
            plt.savefig(os.path.join(directory_for_plots, f"{model_name_to_file}_{label}_{column}_val_f1_scores.png"))
            plt.close()


@click.command()
@click.option(
    "-i",
    "--input-directory-for-results",
    "input_directory_for_results",
    type=click.Path(file_okay=False),
    required=True
)
@click.option(
    "-o",
    "--output-directory-for-plots",
    "output_directory_for_plots",
    type=click.Path(file_okay=False),
    required=True
)
def visualize_all_results(input_directory_for_results: str, output_directory_for_plots: str):
    labels = ['coalitions', 'parties', 'positions']

    visualize_results('KNN', 'knn', os.path.join(input_directory_for_results, 'knn.csv'), labels,
                      output_directory_for_plots, ['distance', 'n_neighbours', 'weights'])
    visualize_results('SVM', 'svm', os.path.join(input_directory_for_results, 'svm.csv'), labels,
                      output_directory_for_plots, ['kernel', 'C', 'tol'])
    visualize_results('Logistic Regression', 'lr', os.path.join(input_directory_for_results, 'logistic_regression.csv')
                      , labels, output_directory_for_plots, ['C', 'tol'])
    visualize_results('Decision Tree', 'dt', os.path.join(input_directory_for_results, 'decision_tree.csv'),
                      labels, output_directory_for_plots, ['min_samples_leaf', 'min_samples_split', 'criterion'])


if __name__ == '__main__':
    visualize_all_results()
