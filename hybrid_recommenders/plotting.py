import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_rating_counts(train_df: pd.DataFrame, group_column: str, title: str, tick_every: int = 25):
    """Long tail plot: number of ratings per item or user, sorted descending."""
    rating_counts = train_df.groupby(group_column)["rating"].count().sort_values(ascending=False)

    rating_counts.plot(kind="bar", xticks=np.arange(0, len(rating_counts), tick_every))
    plt.title(title)
    plt.ylabel("Number of ratings")
    plt.xlabel(group_column)
    plt.tight_layout()
    plt.show()


def plot_svd_tuning(numbers_of_factors: list[int], validation_ndcg: list[float]):
    plt.figure(figsize=(5, 3))
    plt.plot(numbers_of_factors, validation_ndcg, "o-", linewidth=1, markersize=4)
    plt.xscale("log")
    plt.xticks(numbers_of_factors, numbers_of_factors)
    plt.title("SVD tuning on the validation split")
    plt.xlabel("Number of factors")
    plt.ylabel("nDCG@10")
    plt.tight_layout()
    plt.show()


def plot_similarity_heatmap(similarity_table: pd.DataFrame, title: str):
    diagonal_mask = np.eye(similarity_table.shape[0], dtype=bool)

    plt.figure(figsize=(10, 8))
    plt.title(title)
    heatmap_axis = sns.heatmap(similarity_table, annot=True, vmin=0, vmax=1, mask=diagonal_mask, cmap="coolwarm")
    heatmap_axis.set_facecolor("grey")
    plt.show()
