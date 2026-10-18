from collections import defaultdict

import numpy as np
import pandas as pd

# A rating of 3 or higher counts as relevant
RELEVANCE_THRESHOLD = 3.0


def relevant_items_per_user(test_df: pd.DataFrame) -> dict[str, set]:
    relevant_ratings = test_df[test_df["rating"] >= RELEVANCE_THRESHOLD]

    relevant_items = defaultdict(set)
    for user_id, item_id in zip(relevant_ratings["user_id"], relevant_ratings["item_id"]):
        relevant_items[user_id].add(item_id)
    return relevant_items


def hit_rate(recommendations: dict, test_df: pd.DataFrame) -> float:
    """Fraction of users with relevant test items that got at least one of them recommended."""
    relevant_items = relevant_items_per_user(test_df)

    number_of_hits = 0
    for user_id, recommended_items in recommendations.items():
        for item_id, _score in recommended_items:
            if item_id in relevant_items[user_id]:
                number_of_hits += 1
                break

    return number_of_hits / len(relevant_items)


def precision_at_k_per_user(recommendations: dict, test_df: pd.DataFrame, k: int) -> dict[str, float]:
    relevant_items = relevant_items_per_user(test_df)

    precision_per_user = {}
    for user_id in test_df["user_id"].unique():
        precision_per_user[user_id] = 0.0

    for user_id, recommended_items in recommendations.items():
        number_of_relevant_recommendations = 0
        for item_id, _score in recommended_items:
            if item_id in relevant_items[user_id]:
                number_of_relevant_recommendations += 1
        precision_per_user[user_id] = number_of_relevant_recommendations / k

    return precision_per_user


def mean_precision_at_k(recommendations: dict, test_df: pd.DataFrame, k: int) -> float:
    precision_per_user = precision_at_k_per_user(recommendations, test_df, k)
    number_of_test_users = test_df["user_id"].nunique()
    return sum(precision_per_user.values()) / number_of_test_users


def mean_average_precision(recommendations: dict, test_df: pd.DataFrame) -> float:
    """MAP@k, where each user's average precision is normalized by their number of relevant items."""
    relevant_items = relevant_items_per_user(test_df)

    average_precision_per_user = []
    for user_id in relevant_items:
        if user_id not in recommendations:
            average_precision_per_user.append(0.0)

    for user_id, recommended_items in recommendations.items():
        number_of_hits = 0
        sum_of_precisions = 0.0
        for rank, (item_id, _score) in enumerate(recommended_items, start=1):
            if item_id in relevant_items[user_id]:
                number_of_hits += 1
                sum_of_precisions += number_of_hits / rank

        number_of_relevant_items = len(relevant_items[user_id])
        if number_of_relevant_items > 0:
            average_precision_per_user.append(sum_of_precisions / number_of_relevant_items)
        else:
            average_precision_per_user.append(0.0)

    return float(np.mean(average_precision_per_user))


def reciprocal_rank_per_user(recommendations: dict, test_df: pd.DataFrame) -> dict[str, float]:
    """1 / rank of the first relevant recommendation, 0 if there is none."""
    relevant_items = relevant_items_per_user(test_df)

    reciprocal_ranks = {}
    for user_id in relevant_items:
        if user_id not in recommendations:
            reciprocal_ranks[user_id] = 0.0

    for user_id, recommended_items in recommendations.items():
        reciprocal_ranks[user_id] = 0.0
        for rank, (item_id, _score) in enumerate(recommended_items, start=1):
            if item_id in relevant_items[user_id]:
                reciprocal_ranks[user_id] = 1.0 / rank
                break

    return reciprocal_ranks


def mean_reciprocal_rank(recommendations: dict, test_df: pd.DataFrame) -> float:
    reciprocal_ranks = reciprocal_rank_per_user(recommendations, test_df)
    return float(np.mean(list(reciprocal_ranks.values())))


def ndcg_at_k_per_user(recommendations: dict, test_df: pd.DataFrame, k: int) -> dict[str, float]:
    """Binary-gain nDCG@k for every user that has at least one relevant test item."""
    relevant_items = relevant_items_per_user(test_df)

    ndcg_per_user = {}
    for user_id, relevant_items_of_user in relevant_items.items():
        recommended_items = recommendations.get(user_id, [])[:k]

        discounted_cumulative_gain = 0.0
        for rank, (item_id, _score) in enumerate(recommended_items, start=1):
            if item_id in relevant_items_of_user:
                discounted_cumulative_gain += 1.0 / np.log2(rank + 1)

        number_of_ideal_positions = min(k, len(relevant_items_of_user))
        ideal_discounted_cumulative_gain = 0.0
        for rank in range(1, number_of_ideal_positions + 1):
            ideal_discounted_cumulative_gain += 1.0 / np.log2(rank + 1)

        ndcg_per_user[user_id] = discounted_cumulative_gain / ideal_discounted_cumulative_gain

    return ndcg_per_user


def mean_ndcg_at_k(recommendations: dict, test_df: pd.DataFrame, k: int) -> float:
    ndcg_per_user = ndcg_at_k_per_user(recommendations, test_df, k)
    return float(np.mean(list(ndcg_per_user.values())))


def coverage(recommendations: dict, test_df: pd.DataFrame) -> float:
    """Number of distinct recommended items divided by the number of distinct test items.

    Recommended items that are not in the test set are also counted.
    """
    items_in_test_set = set(test_df["item_id"].unique())

    recommended_items = set()
    for user_recommendations in recommendations.values():
        for item_id, _score in user_recommendations:
            recommended_items.add(item_id)

    return len(recommended_items) / len(items_in_test_set)


def evaluate_ranking(recommendations: dict, test_df: pd.DataFrame, k: int = 10) -> dict[str, float]:
    return {
        "HR": hit_rate(recommendations, test_df),
        f"P@{k}": mean_precision_at_k(recommendations, test_df, k),
        f"MAP@{k}": mean_average_precision(recommendations, test_df),
        f"MRR@{k}": mean_reciprocal_rank(recommendations, test_df),
        f"nDCG@{k}": mean_ndcg_at_k(recommendations, test_df, k),
        "Coverage": coverage(recommendations, test_df),
    }


def compare_models(recommendations_per_model: dict[str, dict], test_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    rows = {}
    for model_name, recommendations in recommendations_per_model.items():
        rows[model_name] = evaluate_ranking(recommendations, test_df, k)
    return pd.DataFrame.from_dict(rows, orient="index").round(3)
