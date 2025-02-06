from functools import partial

import lancedb
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from ragifier.config import DatabaseConfig
from ragifier.embed import embed_text


def make_database(model: AutoModel, dataloader: DataLoader, cfg: DatabaseConfig):
    db = lancedb.connect(cfg.path)
    make_batches = partial(embed_text, model=model, dataloader=dataloader)
    tbl = db.create_table(cfg.tbl_name, data=make_batches(), mode="overwrite")
    tbl.create_index(
        num_partitions=cfg.num_partitions,
        num_sub_vectors=cfg.num_sub_vectors,
        vector_column_name="vector",
    )
    tbl.create_scalar_index("id", index_type="BTREE")


def get_centroids(lance_tbl):
    vector_index = lance_tbl.index_statistics("vector_idx")
    centroids = vector_index["indices"][0]["centroids"]
    return torch.tensor(centroids)
