import torch
import torch.nn as nn

# import torch.nn.functional as F


# encourages queries to be far apart and orthogonal
# def query_penalty(queries, temperature):
#     queries = F.normalize(queries, p=2, dim=-1)
#     distance_matrix = torch.cdist(queries, queries, p=2)
#     similarity_matrix = torch.exp(-distance_matrix / temperature)
#     identity = torch.eye(queries.shape[0], device=queries.device)
#     return torch.norm(similarity_matrix - identity, p="fro")


def query_penalty(queries, threshold):
    # Shape: (num_queries, vector_dim)
    normalized_queries = nn.functional.normalize(queries, p=2, dim=1)
    # Shape: (num_queries, num_queries)
    similarity_matrix = torch.matmul(normalized_queries, normalized_queries.T)
    return nn.functional.relu(similarity_matrix - threshold).mean()
