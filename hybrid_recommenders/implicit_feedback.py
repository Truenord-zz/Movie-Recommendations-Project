import numpy as np
import pandas as pd

from hybrid_recommenders.evaluation import RELEVANCE_THRESHOLD

# Recommendations have the usual shape: {user_id: [(item_id, score), ...]}


def build_relevant_matrix(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary user-item matrix of the relevant ratings, with the user id of every row and the item id of every column."""
    user_ids = train_df["user_id"].unique()
    item_ids = train_df["item_id"].unique()

    relevant_ratings = train_df[train_df["rating"] >= RELEVANCE_THRESHOLD]
    relevant_user_rows = pd.Index(user_ids).get_indexer(relevant_ratings["user_id"])
    relevant_item_rows = pd.Index(item_ids).get_indexer(relevant_ratings["item_id"])

    relevant_matrix = np.zeros((len(user_ids), len(item_ids)))
    relevant_matrix[relevant_user_rows, relevant_item_rows] = 1
    return relevant_matrix, user_ids, item_ids


def recommend_top_k_unrated(scores: np.ndarray, train_df: pd.DataFrame, user_ids: np.ndarray,
                            item_ids: np.ndarray, k: int) -> dict[str, list]:
    """The k highest scored items per user, skipping everything the user has rated. Changes `scores` in place."""
    rated_user_rows = pd.Index(user_ids).get_indexer(train_df["user_id"])
    rated_item_rows = pd.Index(item_ids).get_indexer(train_df["item_id"])
    scores[rated_user_rows, rated_item_rows] = -np.inf

    top_item_rows = np.argsort(scores, axis=1)[:, ::-1][:, :k]

    recommendations = {}
    for user_row, user_id in enumerate(user_ids):
        recommendations[user_id] = [(item_ids[item_row], float(scores[user_row, item_row]))
                                    for item_row in top_item_rows[user_row]]
    return recommendations
