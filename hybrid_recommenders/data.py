import pandas as pd


def clean_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing ratings and keep only the last rating per (user, item) pair."""
    cleaned_ratings_df = ratings_df.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    cleaned_ratings_df = cleaned_ratings_df.dropna(subset=["rating"])
    cleaned_ratings_df = cleaned_ratings_df.reset_index(drop=True)
    return cleaned_ratings_df


def load_train_and_test_ratings(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = clean_ratings(pd.read_parquet(train_path))
    test_df = clean_ratings(pd.read_parquet(test_path))

    # Only keep training users that we can also evaluate on
    users_in_test_set = test_df["user_id"].unique()
    train_df = train_df[train_df["user_id"].isin(users_in_test_set)]

    return train_df, test_df


def load_item_metadata(metadata_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Load metadata only for items that appear in the train or test set."""
    items_in_ratings = pd.concat([train_df["item_id"], test_df["item_id"]]).unique()

    all_metadata_item_ids = pd.read_parquet(metadata_path, columns=["item_id"])["item_id"]
    item_ids_to_load = all_metadata_item_ids[all_metadata_item_ids.isin(items_in_ratings)].unique()

    item_metadata_df = pd.read_parquet(
        metadata_path,
        filters=[("item_id", "in", item_ids_to_load.tolist())],
    )
    item_metadata_df = item_metadata_df.reset_index(drop=True)
    return item_metadata_df


def split_most_and_least_active(train_df: pd.DataFrame, group_column: str, fraction: float = 0.2):
    """Return the ids of the top and bottom `fraction` of users or items, ranked by number of ratings."""
    number_of_ratings = train_df.groupby(group_column)["rating"].count().sort_values(ascending=False)
    group_size = int(fraction * train_df[group_column].nunique())

    most_active_ids = number_of_ratings.head(group_size).index
    least_active_ids = number_of_ratings.tail(group_size).index
    return most_active_ids, least_active_ids


def split_off_validation_ratings(train_df: pd.DataFrame, fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a random part of the training ratings for choosing hyperparameters, so the test set stays untouched."""
    validation_df = train_df.sample(frac=fraction, random_state=0)
    fit_df = train_df.drop(validation_df.index)
    return fit_df, validation_df
