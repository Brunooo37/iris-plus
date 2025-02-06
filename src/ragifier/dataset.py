import duckdb
import numpy as np
import polars as pl
import torch
from lancedb.table import Table
from torch.utils.data import Dataset

from ragifier.config import Config


def get_centroids(tbl: Table):
    lance_tbl = tbl.to_lance()  # type: ignore
    vector_index = lance_tbl.index_statistics("vector_idx")
    centroids = vector_index["indices"][0]["centroids"]
    return np.array(centroids)


def get_initial_queries(tbl: Table, vector_dim: int, cfg: Config):
    if cfg.model.query_ini_random:
        return torch.randn(cfg.model.num_queries, vector_dim, dtype=torch.float32)
    else:
        centroids = get_centroids(tbl=tbl)
        # down sample centroids
        indices = np.random.choice(
            np.arange(cfg.database.num_partitions),
            replace=False,
            size=cfg.model.num_queries,
        )
        centroids = centroids[indices]
        return torch.tensor(centroids, dtype=torch.float32)


def get_num_samples(tbl: Table):
    _ = tbl.to_lance()  # type: ignore
    result = duckdb.sql("SELECT COUNT(DISTINCT id) FROM _").fetchone()
    if result:
        n_unique = result[0]
    else:
        raise ValueError("Database is empty.")
    return n_unique


class TableDataset(Dataset):
    def __init__(
        self,
        tbl: Table,
        vector_dim: int,
        k_neighbours: int,
        ignore_index: int,
        initial_queries: np.ndarray,
    ):
        self.tbl = tbl
        self.vector_dim = vector_dim
        self.queries = initial_queries
        self.k_neighbours = k_neighbours
        self.num_samples = get_num_samples(self.tbl)
        self.ignore_index = ignore_index

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        dfs = []
        for query in self.queries:
            df = (
                self.tbl.search(query=query)
                .where(f"id = {idx}", prefilter=True)
                .limit(self.k_neighbours)
                .to_polars()
            )
            dfs.append(df)
        df: pl.DataFrame = pl.concat(dfs)
        if df.height == 0:
            vectors = torch.zeros(1, self.vector_dim)
            label = self.ignore_index
        else:
            vectors = df["vector"].to_torch()
            label = df["label"][0]
        return vectors, torch.tensor(label)
