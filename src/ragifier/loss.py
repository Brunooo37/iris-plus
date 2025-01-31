import torch
import torch.nn.functional as F

# encourages queries to be orthogonal
# def query_penalty(queries, temperature):
#     distance_matrix = torch.cdist(queries, queries, p=2)
#     similarity_matrix = torch.exp(-distance_matrix / temperature)
#     identity = torch.eye(queries.shape[0], device=queries.device)
#     return torch.norm(similarity_matrix - identity, p="fro")


# encourages queries to be close to each other
def query_penalty(queries):
    return torch.cdist(queries, queries, p=2).mean()


# encourages retrieved vectors to be close to the query
def vector_penalty(query, vectors):
    return torch.cdist(vectors, query, p=2).mean()


def loss(outputs, labels, queries, vectors, temperature, alpha, beta):
    # queries = F.normalize(queries, p=2, dim=-1)
    # vectors = F.normalize(vectors, p=2, dim=-1)
    return (
        F.cross_entropy(outputs, labels)
        + alpha * query_penalty(queries)
        + beta * vector_penalty(queries, vectors)
    )
