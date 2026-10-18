"""Pick the number of LightGCN layers on a validation split of the training ratings. The test set is not used."""
import argparse
import os

import numpy as np

from hybrid_recommenders.data import load_train_and_test_ratings, split_off_validation_ratings
from hybrid_recommenders.evaluation import mean_ndcg_at_k
from hybrid_recommenders.lightgcn import recommend_with_lightgcn

NUMBER_OF_EPOCHS = 200
NUMBER_OF_RECOMMENDATIONS = 10
RANDOM_SEEDS = [0, 1, 2]

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def main(data_folder: str, layer_counts: list[int]):
    train_df, _test_df = load_train_and_test_ratings(f"{data_folder}/train_video_games.parquet",
                                                     f"{data_folder}/test_video_games.parquet")

    fit_df, validation_df = split_off_validation_ratings(train_df)

    for number_of_layers in layer_counts:
        ndcg_per_seed = []
        for random_seed in RANDOM_SEEDS:
            recommendations = recommend_with_lightgcn(fit_df, number_of_layers, NUMBER_OF_EPOCHS,
                                                      NUMBER_OF_RECOMMENDATIONS, random_seed)
            ndcg_per_seed.append(mean_ndcg_at_k(recommendations, validation_df, NUMBER_OF_RECOMMENDATIONS))
        print(f"layers={number_of_layers}  nDCG@10={np.mean(ndcg_per_seed):.4f}  std={np.std(ndcg_per_seed):.4f}", flush=True)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data-folder", default=os.path.join(PROJECT_FOLDER, "data"))
    argument_parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    arguments = argument_parser.parse_args()

    main(arguments.data_folder, arguments.layers)
