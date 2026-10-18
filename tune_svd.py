"""Pick the number of SVD factors on a validation split of the training ratings. The test set is not used."""
import argparse
import os

from hybrid_recommenders.collaborative import recommend_with_svd
from hybrid_recommenders.data import load_train_and_test_ratings, split_off_validation_ratings
from hybrid_recommenders.evaluation import mean_ndcg_at_k

NUMBERS_OF_FACTORS = [5, 10, 20, 50, 100, 200]
NUMBER_OF_RECOMMENDATIONS = 10

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def main(data_folder: str):
    train_df, _test_df = load_train_and_test_ratings(f"{data_folder}/train_video_games.parquet",
                                                     f"{data_folder}/test_video_games.parquet")
    fit_df, validation_df = split_off_validation_ratings(train_df)

    for number_of_factors in NUMBERS_OF_FACTORS:
        recommendations = recommend_with_svd(fit_df, number_of_factors, NUMBER_OF_RECOMMENDATIONS)
        ndcg = mean_ndcg_at_k(recommendations, validation_df, NUMBER_OF_RECOMMENDATIONS)
        print(f"factors={number_of_factors}  nDCG@10={ndcg:.4f}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data-folder", default=os.path.join(PROJECT_FOLDER, "data"))
    arguments = argument_parser.parse_args()

    main(arguments.data_folder)
