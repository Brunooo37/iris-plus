import torch
import torch.nn as nn
import torch.nn.functional as F

# encourages queries to be orthogonal
# def query_penalty(queries, temperature):
#     distance_matrix = torch.cdist(queries, queries, p=2)
#     similarity_matrix = torch.exp(-distance_matrix / temperature)
#     identity = torch.eye(queries.shape[0], device=queries.device)
#     return torch.norm(similarity_matrix - identity, p="fro")


# encourages queries to be far apart
def query_penalty(query):
    return torch.cdist(query, query, p=2).mean()


# encourages retrieved vectors to be close to the query
# def vector_penalty(query, vectors):
#     return torch.cdist(query, vectors, p=2).mean()


def loss(outputs, labels, queries, alpha):  # , beta, vectors,
    queries = F.normalize(queries, p=2, dim=-1)
    # vectors = F.normalize(vectors, p=2, dim=-1)
    return (
        F.cross_entropy(outputs, labels) + alpha * query_penalty(queries)
        # + beta * vector_penalty(query, vectors)
    )


def retrival_loss(predictions, targets, queries, lambda_penalty, threshold):
    """
    BCE loss + penalty term for query vector similarities
    Args:
        predictions: model predictions
        targets: true labels
        queries: model query vectors, tensor of shape (num_queries, vector_dim)
        lambda_penalty: hyperparamter for controling the penalty term
        threshold: penalty threshold, we only penalize cos-smilarities > threshold
    Returns:
        top k chunk embeddings: tensor of shape (batch, k, vector_dim)
    """
    bce_loss = nn.functional.binary_cross_entropy(predictions, targets)
    normalized_queries = nn.functional.normalize(
        queries, p=2, dim=1
    )  # Shape: (num_queries, vector_dim)
    similarity_matrix = torch.matmul(
        normalized_queries, normalized_queries.T
    )  # Shape: (num_queries, num_queries)
    penalty = nn.functional.relu(similarity_matrix - threshold).mean()
    total_loss = bce_loss + lambda_penalty * penalty
    return total_loss
