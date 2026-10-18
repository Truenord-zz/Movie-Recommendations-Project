"""Tune RP3beta and GF-CF on a validation split of the training ratings. The test set is not used."""
import argparse
import os

from hybrid_recommenders.data import load_train_and_test_ratings, split_off_validation_ratings
from hybrid_recommenders.evaluation import mean_ndcg_at_k
from hybrid_recommenders.graph_methods import recommend_with_gf_cf, recommend_with_rp3_beta

RP3_BETA_ALPHAS = [0.25, 0.5, 0.75, 1.0]
RP3_BETA_BETAS = [0.25, 0.5, 0.75]
GF_CF_NUMBERS_OF_SINGULAR_VECTORS = [4, 8, 16, 32, 64, 128, 256]
GF_CF_ALPHAS = [0.3, 1.0]
NUMBER_OF_RECOMMENDATIONS = 10

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def main(data_folder: str):
    train_df, _test_df = load_train_and_test_ratings(f"{data_folder}/train_video_games.parquet",
                                                     f"{data_folder}/test_video_games.parquet")
    fit_df, validation_df = split_off_validation_ratings(train_df)

    for alpha in RP3_BETA_ALPHAS:
        for beta in RP3_BETA_BETAS:
            recommendations = recommend_with_rp3_beta(fit_df, alpha, beta, NUMBER_OF_RECOMMENDATIONS)
            ndcg = mean_ndcg_at_k(recommendations, validation_df, NUMBER_OF_RECOMMENDATIONS)
            print(f"RP3beta  alpha={alpha}  beta={beta}  nDCG@10={ndcg:.4f}")

    for number_of_singular_vectors in GF_CF_NUMBERS_OF_SINGULAR_VECTORS:
        for alpha in GF_CF_ALPHAS:
            recommendations = recommend_with_gf_cf(fit_df, number_of_singular_vectors, alpha, NUMBER_OF_RECOMMENDATIONS)
            ndcg = mean_ndcg_at_k(recommendations, validation_df, NUMBER_OF_RECOMMENDATIONS)
            print(f"GF-CF  singular_vectors={number_of_singular_vectors}  alpha={alpha}  nDCG@10={ndcg:.4f}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data-folder", default=os.path.join(PROJECT_FOLDER, "data"))
    arguments = argument_parser.parse_args()

    main(arguments.data_folder)
