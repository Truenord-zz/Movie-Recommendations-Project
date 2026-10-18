import os

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fit_tfidf(corpus: pd.Series):
    # The corpus is already lowercased and tokenized, so we only split on whitespace
    vectorizer = TfidfVectorizer(lowercase=False, analyzer="word", token_pattern=r"\S+")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer


def glove_embeddings(texts: pd.Series, glove_model_name: str = "glove-wiki-gigaword-100") -> np.ndarray:
    """Each text becomes the mean of the GloVe vectors of its words."""
    import gensim.downloader

    glove_model = gensim.downloader.load(glove_model_name)

    embeddings = []
    for text in texts:
        word_vectors = [glove_model[word] for word in text.split() if word in glove_model]
        if len(word_vectors) == 0:
            embeddings.append(np.zeros(glove_model.vector_size))
        else:
            embeddings.append(np.mean(word_vectors, axis=0))

    return np.array(embeddings)


def transformer_embeddings(texts: pd.Series, model_name: str, batch_size: int = 1) -> np.ndarray:
    """Mean of the last hidden states over the non-padding tokens.

    batch_size is 1 because larger batches ran out of memory on my M1.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    texts = texts.tolist()
    embeddings = []
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        encoded_batch = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            last_hidden_states = model(**encoded_batch).last_hidden_state

        attention_mask = encoded_batch["attention_mask"].unsqueeze(-1)
        sum_of_token_embeddings = (last_hidden_states * attention_mask).sum(dim=1)
        number_of_tokens = attention_mask.sum(dim=1)
        mean_embeddings = sum_of_token_embeddings / number_of_tokens
        embeddings.append(mean_embeddings.cpu())

        if device == "mps":
            torch.mps.empty_cache()

    return torch.cat(embeddings).numpy()


def load_or_compute_embeddings(cache_path: str, compute_embeddings) -> np.ndarray:
    """The transformer embeddings take a long time, so we only compute them once."""
    if os.path.exists(cache_path):
        return np.load(cache_path)

    embeddings = compute_embeddings()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def build_user_profiles(item_vectors, item_ids: list, train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """A user profile is the mean of the user's rated item vectors, weighted by rating."""
    row_of_item = {item_id: row for row, item_id in enumerate(item_ids)}

    user_profiles = {}
    for user_id, user_ratings in train_df.groupby("user_id"):
        user_ratings = user_ratings[user_ratings["item_id"].isin(row_of_item)]
        if len(user_ratings) == 0:
            continue

        item_rows = [row_of_item[item_id] for item_id in user_ratings["item_id"]]
        rated_item_vectors = item_vectors[item_rows]
        if scipy.sparse.issparse(rated_item_vectors):
            rated_item_vectors = rated_item_vectors.toarray()

        rating_weights = user_ratings["rating"].to_numpy(dtype=float)
        rating_weights = rating_weights / rating_weights.sum()

        weighted_item_vectors = rated_item_vectors * rating_weights.reshape(-1, 1)
        user_profiles[user_id] = weighted_item_vectors.sum(axis=0).reshape(1, -1)

    return user_profiles


def recommend_by_profile_similarity(item_vectors, item_ids: list, train_df: pd.DataFrame, k: int = 10) -> dict[str, list]:
    """Recommend the k unrated items most similar to each user's profile."""
    user_profiles = build_user_profiles(item_vectors, item_ids, train_df)
    rated_items_per_user = train_df.groupby("user_id")["item_id"].apply(set)

    recommendations = {}
    for user_id, user_profile in user_profiles.items():
        similarity_scores = cosine_similarity(user_profile, item_vectors)[0]
        rows_by_descending_similarity = np.argsort(similarity_scores)[::-1]

        user_recommendations = []
        for row in rows_by_descending_similarity:
            item_id = item_ids[row]
            if item_id not in rated_items_per_user[user_id]:
                user_recommendations.append((item_id, float(similarity_scores[row])))
            if len(user_recommendations) == k:
                break

        recommendations[user_id] = user_recommendations

    return recommendations


def item_similarity_table(item_vectors, item_ids: list, selected_item_ids: list, item_titles: dict) -> pd.DataFrame:
    """Cosine similarity between a hand-picked set of items, labelled by (shortened) title."""
    row_of_item = {item_id: row for row, item_id in enumerate(item_ids)}
    selected_rows = [row_of_item[item_id] for item_id in selected_item_ids]

    similarity_matrix = np.round(cosine_similarity(item_vectors[selected_rows]), 3)

    short_titles = [item_titles[item_id][:20] for item_id in selected_item_ids]
    return pd.DataFrame(similarity_matrix, index=short_titles, columns=short_titles)
