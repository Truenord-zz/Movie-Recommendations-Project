import numpy as np
import pandas as pd

from hybrid_recommenders.implicit_feedback import build_relevant_matrix, recommend_top_k_unrated

# Two graph methods that need no training, both work on the bipartite graph of users and the items they rated


def recommend_with_rp3_beta(train_df: pd.DataFrame, alpha: float, beta: float, k: int) -> dict[str, list]:
    """RP3beta: item to item similarity from random walks of length two (item -> user -> item).

    The walk probabilities are raised to the power alpha, and the similarity to an item is
    divided by its popularity raised to the power beta, so popular items are less dominant.
    """
    relevant_matrix, user_ids, item_ids = build_relevant_matrix(train_df)

    number_of_items_per_user = relevant_matrix.sum(axis=1).clip(min=1)
    number_of_users_per_item = relevant_matrix.sum(axis=0).clip(min=1)

    user_to_item_probabilities = relevant_matrix / number_of_items_per_user[:, None]
    item_to_user_probabilities = relevant_matrix.T / number_of_users_per_item[:, None]

    item_similarities = (item_to_user_probabilities ** alpha) @ (user_to_item_probabilities ** alpha)
    item_similarities = item_similarities / (number_of_users_per_item ** beta)
    np.fill_diagonal(item_similarities, 0)

    scores = relevant_matrix @ item_similarities
    return recommend_top_k_unrated(scores, train_df, user_ids, item_ids, k)


def recommend_with_gf_cf(train_df: pd.DataFrame, number_of_singular_vectors: int, alpha: float,
                         k: int) -> dict[str, list]:
    """GF-CF: graph filtering with a linear filter plus an ideal low-pass filter.

    The ideal low-pass filter keeps the strongest singular vectors of the normalised user-item matrix,
    and alpha is how much it counts compared to the linear filter.
    """
    relevant_matrix, user_ids, item_ids = build_relevant_matrix(train_df)

    number_of_items_per_user = relevant_matrix.sum(axis=1).clip(min=1)
    number_of_users_per_item = relevant_matrix.sum(axis=0).clip(min=1)
    inverse_sqrt_user_degrees = number_of_items_per_user ** -0.5
    inverse_sqrt_item_degrees = number_of_users_per_item ** -0.5
    sqrt_item_degrees = number_of_users_per_item ** 0.5

    normalized_matrix = relevant_matrix * inverse_sqrt_user_degrees[:, None] * inverse_sqrt_item_degrees[None, :]
    linear_filter = normalized_matrix.T @ normalized_matrix

    _, _, transposed_item_vectors = np.linalg.svd(normalized_matrix, full_matrices=False)
    strongest_item_vectors = transposed_item_vectors[:number_of_singular_vectors].T
    low_pass_filter = strongest_item_vectors @ strongest_item_vectors.T
    low_pass_filter = inverse_sqrt_item_degrees[:, None] * low_pass_filter * sqrt_item_degrees[None, :]

    scores = relevant_matrix @ (linear_filter + alpha * low_pass_filter)
    return recommend_top_k_unrated(scores, train_df, user_ids, item_ids, k)
