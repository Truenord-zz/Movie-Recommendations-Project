"""run it all, and save to a file"""
import argparse
import os

from hybrid_recommenders.collaborative import (
    recommend_most_popular,
    recommend_with_item_knn,
    recommend_with_svd,
)
from hybrid_recommenders.content_based import (
    fit_tfidf,
    glove_embeddings,
    load_or_compute_embeddings,
    recommend_by_profile_similarity,
    transformer_embeddings,
)
from hybrid_recommenders.data import load_item_metadata, load_train_and_test_ratings
from hybrid_recommenders.evaluation import compare_models
from hybrid_recommenders.graph_methods import recommend_with_gf_cf, recommend_with_rp3_beta
from hybrid_recommenders.hybrids import (
    add_content_based_ratings_for_sparse_users,
    reciprocal_rank_fusion,
    switching_hybrid,
)
from hybrid_recommenders.lightgcn import recommend_with_lightgcn
from hybrid_recommenders.text_preprocessing import build_item_corpus

# Chosen by tune_svd.py on a validation split of the training ratings
SVD_NUMBER_OF_FACTORS = 20

# Chosen by tune_lightgcn.py on a validation split of the training ratings
LIGHTGCN_NUMBER_OF_LAYERS = 3
LIGHTGCN_NUMBER_OF_EPOCHS = 200

# Chosen by tune_graph_methods.py on a validation split of the training ratings
RP3_BETA_ALPHA = 0.5
RP3_BETA_BETA = 0.5
GF_CF_NUMBER_OF_SINGULAR_VECTORS = 16
GF_CF_ALPHA = 0.3

NUMBER_OF_RECOMMENDATIONS = 10

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
CACHE_FOLDER = os.path.join(PROJECT_FOLDER, "cache")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")


def collaborative_recommendations(train_df):
    knn_recommendations = recommend_with_item_knn(train_df, NUMBER_OF_RECOMMENDATIONS)
    svd_recommendations = recommend_with_svd(train_df, SVD_NUMBER_OF_FACTORS, NUMBER_OF_RECOMMENDATIONS)
    return knn_recommendations, svd_recommendations


def recommendations_from_all_models(train_df, test_df, item_metadata_df, include_embeddings: bool) -> dict:
    item_ids = item_metadata_df["item_id"].tolist()
    recommendations_per_model = {}

    recommendations_per_model["TopPop"] = recommend_most_popular(
        train_df, test_df["user_id"].unique(), NUMBER_OF_RECOMMENDATIONS
    )
    knn_recommendations, svd_recommendations = collaborative_recommendations(train_df)
    recommendations_per_model["KNN"] = knn_recommendations
    recommendations_per_model["SVD"] = svd_recommendations
    recommendations_per_model["LightGCN"] = recommend_with_lightgcn(
        train_df, LIGHTGCN_NUMBER_OF_LAYERS, LIGHTGCN_NUMBER_OF_EPOCHS, NUMBER_OF_RECOMMENDATIONS
    )
    # BPR-MF is LightGCN without any layers, so the graph is not used
    recommendations_per_model["BPR-MF"] = recommend_with_lightgcn(
        train_df, 0, LIGHTGCN_NUMBER_OF_EPOCHS, NUMBER_OF_RECOMMENDATIONS
    )
    recommendations_per_model["RP3beta"] = recommend_with_rp3_beta(
        train_df, RP3_BETA_ALPHA, RP3_BETA_BETA, NUMBER_OF_RECOMMENDATIONS
    )
    recommendations_per_model["GF-CF"] = recommend_with_gf_cf(
        train_df, GF_CF_NUMBER_OF_SINGULAR_VECTORS, GF_CF_ALPHA, NUMBER_OF_RECOMMENDATIONS
    )

    item_corpus = build_item_corpus(item_metadata_df)
    tfidf_matrix, _vectorizer = fit_tfidf(item_corpus)
    tfidf_recommendations = recommend_by_profile_similarity(tfidf_matrix, item_ids, train_df, NUMBER_OF_RECOMMENDATIONS)
    recommendations_per_model["CB TF-IDF"] = tfidf_recommendations

    if include_embeddings:
        embedding_functions = {
            "CB GloVe": lambda: glove_embeddings(item_corpus),
            "CB BERT": lambda: transformer_embeddings(item_corpus, "bert-base-uncased"),
            "CB RoBERTa": lambda: transformer_embeddings(item_corpus, "roberta-base"),
        }
        for model_name, compute_embeddings in embedding_functions.items():
            cache_file_name = model_name.replace(" ", "_").lower() + "_corpus.npy"
            cache_path = os.path.join(CACHE_FOLDER, cache_file_name)
            item_embeddings = load_or_compute_embeddings(cache_path, compute_embeddings)
            recommendations_per_model[model_name] = recommend_by_profile_similarity(
                item_embeddings, item_ids, train_df, NUMBER_OF_RECOMMENDATIONS
            )

    # All hybrids combine collaborative filtering with the TF-IDF content-based model
    recommendations_per_model["Parallel RRF (KNN+CB)"] = reciprocal_rank_fusion(knn_recommendations, tfidf_recommendations)
    recommendations_per_model["Parallel RRF (SVD+CB)"] = reciprocal_rank_fusion(svd_recommendations, tfidf_recommendations)
    recommendations_per_model["Switching (KNN+CB)"] = switching_hybrid(knn_recommendations, tfidf_recommendations, train_df)
    recommendations_per_model["Switching (SVD+CB)"] = switching_hybrid(svd_recommendations, tfidf_recommendations, train_df)

    enriched_train_df = add_content_based_ratings_for_sparse_users(train_df, tfidf_recommendations)
    sequential_knn_recommendations, sequential_svd_recommendations = collaborative_recommendations(enriched_train_df)
    recommendations_per_model["Sequential (CB -> KNN)"] = sequential_knn_recommendations
    recommendations_per_model["Sequential (CB -> SVD)"] = sequential_svd_recommendations

    return recommendations_per_model


def main(data_folder: str, include_embeddings: bool):
    train_df, test_df = load_train_and_test_ratings(f"{data_folder}/train_video_games.parquet",
                                                    f"{data_folder}/test_video_games.parquet")
    item_metadata_df = load_item_metadata(f"{data_folder}/meta_video_games.parquet", train_df, test_df)

    recommendations_per_model = recommendations_from_all_models(train_df, test_df, item_metadata_df, include_embeddings)

    comparison_table = compare_models(recommendations_per_model, test_df, NUMBER_OF_RECOMMENDATIONS)
    print(comparison_table.to_string())
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    comparison_table.to_csv(os.path.join(RESULTS_FOLDER, "model_comparison.csv"))


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data-folder", default=os.path.join(PROJECT_FOLDER, "data"))
    argument_parser.add_argument("--skip-embeddings", action="store_true", help="Skip GloVe, BERT and RoBERTa")
    arguments = argument_parser.parse_args()

    main(arguments.data_folder, include_embeddings=not arguments.skip_embeddings)
