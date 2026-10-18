import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_recommenders.evaluation import RELEVANCE_THRESHOLD

# Recommendations have the usual shape: {user_id: [(item_id, score), ...]}

EMBEDDING_SIZE = 64
BATCH_SIZE = 2048
REGULARIZATION_WEIGHT = 1e-4


def build_normalized_adjacency(user_rows: torch.Tensor, item_rows: torch.Tensor,
                               number_of_users: int, number_of_items: int) -> torch.Tensor:
    """Adjacency matrix of the user-item graph, normalised by the degrees of both ends of every edge.

    The users are the first nodes and the items come after them.
    """
    number_of_nodes = number_of_users + number_of_items
    item_nodes = item_rows + number_of_users

    start_nodes = torch.cat([user_rows, item_nodes])
    end_nodes = torch.cat([item_nodes, user_rows])

    number_of_neighbours = torch.bincount(start_nodes, minlength=number_of_nodes)
    inverse_sqrt_neighbours = number_of_neighbours.clamp(min=1).float().pow(-0.5)
    edge_weights = inverse_sqrt_neighbours[start_nodes] * inverse_sqrt_neighbours[end_nodes]

    edge_indices = torch.stack([start_nodes, end_nodes])
    return torch.sparse_coo_tensor(edge_indices, edge_weights, (number_of_nodes, number_of_nodes),
                                   check_invariants=True)


class LightGCN(nn.Module):
    def __init__(self, adjacency: torch.Tensor, number_of_users: int, number_of_layers: int):
        super().__init__()
        self.adjacency = adjacency
        self.number_of_users = number_of_users
        self.number_of_layers = number_of_layers

        self.node_embeddings = nn.Embedding(adjacency.shape[0], EMBEDDING_SIZE)
        nn.init.normal_(self.node_embeddings.weight, std=0.1)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Final embedding of a node is the mean of its embeddings from layer 0 up to the last layer."""
        layer_embeddings = self.node_embeddings.weight
        embeddings_from_all_layers = [layer_embeddings]
        for _ in range(self.number_of_layers):
            layer_embeddings = torch.sparse.mm(self.adjacency, layer_embeddings)
            embeddings_from_all_layers.append(layer_embeddings)

        mean_embeddings = torch.stack(embeddings_from_all_layers).mean(dim=0)
        return mean_embeddings[:self.number_of_users], mean_embeddings[self.number_of_users:]


def train_with_bpr(model: LightGCN, user_rows: torch.Tensor, item_rows: torch.Tensor,
                   number_of_epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    number_of_items = model.adjacency.shape[0] - model.number_of_users

    for _ in range(number_of_epochs):
        shuffled_edges = torch.randperm(len(user_rows))
        for batch_start in range(0, len(shuffled_edges), BATCH_SIZE):
            batch_edges = shuffled_edges[batch_start:batch_start + BATCH_SIZE]
            batch_users = user_rows[batch_edges]
            positive_items = item_rows[batch_edges]
            negative_items = torch.randint(number_of_items, (len(batch_edges),))

            user_embeddings, item_embeddings = model.propagate()
            positive_scores = (user_embeddings[batch_users] * item_embeddings[positive_items]).sum(dim=1)
            negative_scores = (user_embeddings[batch_users] * item_embeddings[negative_items]).sum(dim=1)
            bpr_loss = -F.logsigmoid(positive_scores - negative_scores).mean()

            # Only the layer 0 embeddings are parameters, so they are the ones that get regularized
            layer_zero_embeddings = model.node_embeddings.weight
            batch_layer_zero = (layer_zero_embeddings[batch_users].pow(2).sum()
                                + layer_zero_embeddings[positive_items + model.number_of_users].pow(2).sum()
                                + layer_zero_embeddings[negative_items + model.number_of_users].pow(2).sum())
            regularization = batch_layer_zero / (2 * len(batch_edges))

            loss = bpr_loss + REGULARIZATION_WEIGHT * regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def recommend_with_lightgcn(train_df: pd.DataFrame, number_of_layers: int, number_of_epochs: int,
                            k: int, random_seed: int = 0) -> dict[str, list]:
    """Train LightGCN on the relevant ratings and recommend the k highest scored items the user has not rated."""
    torch.manual_seed(random_seed)

    user_ids = train_df["user_id"].unique()
    item_ids = train_df["item_id"].unique()
    row_of_user = {user_id: row for row, user_id in enumerate(user_ids)}
    row_of_item = {item_id: row for row, item_id in enumerate(item_ids)}

    relevant_ratings = train_df[train_df["rating"] >= RELEVANCE_THRESHOLD]
    relevant_user_rows = torch.tensor(relevant_ratings["user_id"].map(row_of_user).to_numpy())
    relevant_item_rows = torch.tensor(relevant_ratings["item_id"].map(row_of_item).to_numpy())

    adjacency = build_normalized_adjacency(relevant_user_rows, relevant_item_rows, len(user_ids), len(item_ids))
    model = LightGCN(adjacency, len(user_ids), number_of_layers)
    train_with_bpr(model, relevant_user_rows, relevant_item_rows, number_of_epochs)

    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate()
        scores = user_embeddings @ item_embeddings.T

    rated_user_rows = torch.tensor(train_df["user_id"].map(row_of_user).to_numpy())
    rated_item_rows = torch.tensor(train_df["item_id"].map(row_of_item).to_numpy())
    scores[rated_user_rows, rated_item_rows] = float("-inf")

    top_scores, top_item_rows = torch.topk(scores, k)

    recommendations = {}
    for user_id, user_row in row_of_user.items():
        recommended_item_ids = item_ids[top_item_rows[user_row].numpy()]
        recommended_scores = top_scores[user_row].tolist()
        recommendations[user_id] = list(zip(recommended_item_ids, recommended_scores))
    return recommendations
