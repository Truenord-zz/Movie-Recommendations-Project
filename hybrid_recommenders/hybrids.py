from collections import defaultdict

import pandas as pd

# Users with at most this many training ratings count as sparse (the bottom 50%)
SPARSE_USER_RATING_THRESHOLD = 16


def reciprocal_rank_fusion(collaborative_recommendations: dict, content_recommendations: dict,
                           kappa: int = 60, k: int = 10) -> dict[str, list]:
    """Parallel hybrid: score(item) = sum over both lists of 1 / (kappa + rank)."""
    all_users = set(collaborative_recommendations) | set(content_recommendations)

    fused_recommendations = {}
    for user_id in all_users:
        fused_score_per_item = defaultdict(float)

        for rank, (item_id, _score) in enumerate(collaborative_recommendations.get(user_id, []), start=1):
            fused_score_per_item[item_id] += 1 / (kappa + rank)

        for rank, (item_id, _score) in enumerate(content_recommendations.get(user_id, []), start=1):
            fused_score_per_item[item_id] += 1 / (kappa + rank)

        sorted_by_fused_score = sorted(fused_score_per_item.items(), key=lambda item_and_score: item_and_score[1], reverse=True)
        fused_recommendations[user_id] = sorted_by_fused_score[:k]

    return fused_recommendations


def switching_hybrid(collaborative_recommendations: dict, content_recommendations: dict,
                     train_df: pd.DataFrame, k: int = 10) -> dict[str, list]:
    """Content-based for sparse users, collaborative filtering for everyone else."""
    number_of_ratings_per_user = train_df.groupby("user_id")["item_id"].count()
    all_users = set(collaborative_recommendations) | set(content_recommendations)

    switched_recommendations = {}
    for user_id in all_users:
        number_of_ratings = number_of_ratings_per_user.get(user_id, 0)

        if number_of_ratings <= SPARSE_USER_RATING_THRESHOLD:
            switched_recommendations[user_id] = content_recommendations.get(user_id, [])[:k]
        else:
            switched_recommendations[user_id] = collaborative_recommendations.get(user_id, [])[:k]

    return switched_recommendations


def add_content_based_ratings_for_sparse_users(train_df: pd.DataFrame, content_recommendations: dict) -> pd.DataFrame:
    """Sequential hybrid, step 1: add synthetic ratings from the content-based recommender to sparse users.

    The collaborative model is then trained on the enriched ratings.
    """
    synthetic_ratings = []
    for user_id, user_ratings in train_df.groupby("user_id"):
        if user_ratings["item_id"].nunique() > SPARSE_USER_RATING_THRESHOLD:
            continue

        for item_id, cosine_score in content_recommendations[user_id]:
            # Cosine similarity is between 0 and 1, ratings are between 1 and 5
            synthetic_rating = cosine_score * 5
            synthetic_ratings.append({"user_id": user_id, "item_id": item_id, "rating": synthetic_rating})

    synthetic_ratings_df = pd.DataFrame(synthetic_ratings)
    enriched_train_df = pd.concat([train_df, synthetic_ratings_df], ignore_index=True)
    return enriched_train_df
