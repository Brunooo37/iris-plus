from dataclasses import dataclass
from functools import partial

import duckdb
import numpy as np
import polars as pl
import torch
from lancedb.table import LanceTable
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from iris.config import Config


def get_centroids(tbl: LanceTable) -> np.ndarray:
    lance_tbl = tbl.to_lance()
    vector_index = lance_tbl.index_statistics("vector_idx")
    centroids = vector_index["indices"][0]["centroids"]
    return np.array(centroids)


def get_ini_queries(tbl: LanceTable, cfg: Config) -> np.ndarray:
    if cfg.model.query_ini_random:
        return np.random.randn(cfg.model.num_queries, cfg.model.d_model)
    else:
        centroids = get_centroids(tbl=tbl)
        # down sample centroids to number of queries
        indices = np.random.choice(
            np.arange(cfg.database.num_partitions),
            replace=False,
            size=cfg.model.num_queries,
        )
        return centroids[indices]


def get_num_classes(tbl: LanceTable):
    lance_tbl = tbl.to_lance()
    duckdb.register("lance_tbl", lance_tbl)
    num_labels = duckdb.sql("SELECT COUNT(DISTINCT label) FROM lance_tbl").fetchone()
    return num_labels[0] if num_labels else 0


class LanceTableDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        tbl: LanceTable,
        vector_dim: int,
        k_neighbors: int,
        ignore_index: int,
        ini_queries: np.ndarray,
    ):
        self.ids = ids
        self.tbl = tbl
        self.vector_dim = vector_dim
        self.k_neighbors = k_neighbors
        self.ignore_index = ignore_index
        self.queries = ini_queries

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        dfs: list[pl.DataFrame] = []
        id = self.ids[idx]
        chunk_ids = []
        for query in self.queries:
            condition = f"(id = {id})"
            if chunk_ids:
                excluded_chunks = ", ".join(chunk_ids)
                condition += f" AND (chunk_id NOT IN ({excluded_chunks}))"
            df = (
                self.tbl.search(query=query)
                .where(condition, prefilter=True)
                .limit(self.k_neighbors)
                .to_polars()
            )
            chunk_ids.extend(df["chunk_id"].cast(pl.String).to_list())
            dfs.append(df)
        df = pl.concat(dfs)
        if df.height == 0:
            vectors = torch.randn(1, self.vector_dim)
            label = self.ignore_index
        else:
            vectors = df["vector"].to_torch()
            label = df["label"][0]
        return vectors, torch.tensor(label)


def collate_fn(batch):
    vectors, labels = zip(*batch)
    vectors = pad_sequence(vectors, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    attention_mask = (vectors != 0).all(dim=-1)
    return vectors, labels, attention_mask


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def train_val_test_split(data: np.ndarray, seed: int):
    train, temp = train_test_split(data, test_size=0.2, shuffle=True, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    return train, val, test


def make_loaders(cfg: Config, tbl: LanceTable):
    lance_tbl = tbl.to_lance()
    duckdb.register("lance_tbl", lance_tbl)
    df = duckdb.sql("SELECT DISTINCT id FROM lance_tbl").to_df()
    ids = df["id"].to_numpy()
    train, val, test = train_val_test_split(ids, seed=cfg.seed)
    ini_queries = get_ini_queries(tbl=tbl, cfg=cfg)
    dataset = partial(
        LanceTableDataset,
        tbl=tbl,
        vector_dim=cfg.model.d_model,
        k_neighbors=cfg.database.k_neighbors,
        ignore_index=cfg.trainer.ignore_index,
        ini_queries=ini_queries,
    )
    train = dataset(ids=train)
    val = dataset(ids=val)
    test = dataset(ids=test)
    loader = partial(DataLoader, **cfg.dataloader.model_dump(), collate_fn=collate_fn)
    train_loader = loader(train, shuffle=True, drop_last=True)
    val_loader = loader(val)
    test_loader = loader(test)
    return DataLoaders(train=train_loader, validation=val_loader, test=test_loader)
