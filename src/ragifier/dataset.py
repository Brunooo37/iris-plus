from functools import partial

import duckdb
import numpy as np
import polars as pl
import torch
from lancedb.table import Table
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

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
        # down sample centroids to number of queries
        indices = np.random.choice(
            np.arange(cfg.database.num_partitions),
            replace=False,
            size=cfg.model.num_queries,
        )
        centroids = centroids[indices]
        return torch.tensor(centroids, dtype=torch.float32)


class TableDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        tbl: Table,
        vector_dim: int,
        k_neighbors: int,
        ignore_index: int,
        initial_queries: np.ndarray,
    ):
        self.ids = ids
        self.tbl = tbl
        self.vector_dim = vector_dim
        self.k_neighbors = k_neighbors
        self.ignore_index = ignore_index
        self.queries = initial_queries

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        dfs = []
        id = self.ids[idx]
        for query in self.queries:
            df = (
                self.tbl.search(query=query)
                .where(f"id = {id}", prefilter=True)
                .limit(self.k_neighbors)
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


def collate_fn(batch):
    vectors, labels = zip(*batch)
    vectors = pad_sequence(vectors, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    attention_mask = (vectors == 0).all(dim=-1)
    return vectors, labels, attention_mask


def make_dataloaders(
    tbl: Table, vector_dim: int, initial_queries: torch.Tensor, cfg: Config
):
    _ = tbl.to_lance()  # type: ignore
    df = duckdb.sql("SELECT DISTINCT id FROM _").to_df()
    ids = df["id"].to_numpy()
    train, temp = train_test_split(
        ids, test_size=0.2, shuffle=True, random_state=cfg.seed
    )
    val, test = train_test_split(temp, test_size=0.5, random_state=cfg.seed)
    queries = initial_queries.numpy()
    dataset = partial(
        TableDataset,
        tbl=tbl,
        vector_dim=vector_dim,
        k_neighbors=cfg.model.k_neighbors,
        ignore_index=cfg.ignore_index,
        initial_queries=queries,
    )
    train = dataset(ids=train)
    val = dataset(ids=val)
    test = dataset(ids=test)
    loader = partial(DataLoader, **cfg.dataloader.model_dump(), collate_fn=collate_fn)
    train_loader = loader(train, shuffle=True)
    val_loader = loader(val)
    test_loader = loader(test)
    return train_loader, val_loader, test_loader
