import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from hybrid_recommenders.evaluation import RELEVANCE_THRESHOLD
from hybrid_recommenders.implicit_feedback import build_relevant_matrix, recommend_top_k_unrated

# Recommendations everywhere in the project have this shape:
# {user_id: [(item_id, score), ...]} sorted by score, highest first


def recommend_most_popular(train_df: pd.DataFrame, users_to_recommend_for, k: int) -> dict[str, list]:
    """TopPop baseline: the most popular items the user has not rated, ordered by their average relevant rating."""
    relevant_ratings = train_df[train_df["rating"] >= RELEVANCE_THRESHOLD]
    popularity_per_item = relevant_ratings.groupby("item_id")["rating"].agg(["count", "mean"])
    items_by_popularity = popularity_per_item.sort_values("count", ascending=False)

    rated_items_per_user = train_df.groupby("user_id")["item_id"].apply(set)

    recommendations = {}
    for user_id in users_to_recommend_for:
        rated_items = rated_items_per_user.get(user_id, set())
        unrated_items_by_popularity = items_by_popularity[~items_by_popularity.index.isin(rated_items)]
        most_popular_unrated_items = unrated_items_by_popularity.head(k)

        average_rating_per_item = most_popular_unrated_items["mean"].sort_values(ascending=False)
        recommendations[user_id] = list(average_rating_per_item.items())
    return recommendations


def recommend_with_item_knn(train_df: pd.DataFrame, k: int) -> dict[str, list]:
    """Item-based KNN on implicit feedback.

    An item is scored by its summed cosine similarity to the relevant items the user has rated.
    Using all items as neighbours was better than keeping only the nearest ones on a validation split.
    """
    relevant_matrix, user_ids, item_ids = build_relevant_matrix(train_df)

    item_similarities = cosine_similarity(relevant_matrix.T)
    np.fill_diagonal(item_similarities, 0)

    scores = relevant_matrix @ item_similarities
    return recommend_top_k_unrated(scores, train_df, user_ids, item_ids, k)


def recommend_with_svd(train_df: pd.DataFrame, number_of_factors: int, k: int) -> dict[str, list]:
    """SVD of the binary relevant matrix: an item is scored by how well the strongest factors reconstruct it."""
    relevant_matrix, user_ids, item_ids = build_relevant_matrix(train_df)

    _, _, transposed_item_vectors = np.linalg.svd(relevant_matrix, full_matrices=False)
    strongest_item_vectors = transposed_item_vectors[:number_of_factors].T

    scores = relevant_matrix @ strongest_item_vectors @ strongest_item_vectors.T
    return recommend_top_k_unrated(scores, train_df, user_ids, item_ids, k)
