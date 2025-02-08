from functools import partial

import lancedb
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from rassifier.config import Config
from rassifier.data import get_dataset
from rassifier.encode import make_batches


def make_database(model: AutoModel, cfg: Config):
    dataset = get_dataset(cfg=cfg)
    dataloader = DataLoader(dataset, **cfg.dataloader.model_dump())  # type: ignore
    db = lancedb.connect(cfg.database.path)
    batch_fn = partial(make_batches, model=model, dataloader=dataloader)
    tbl = db.create_table(cfg.database.tbl_name, data=batch_fn(), mode="overwrite")
    tbl.create_index(
        num_partitions=cfg.database.num_partitions,
        num_sub_vectors=cfg.database.num_sub_vectors,
    )
    tbl.create_scalar_index("id", index_type="BTREE")


def get_centroids(lance_tbl):
    vector_index = lance_tbl.index_statistics("vector_idx")
    centroids = vector_index["indices"][0]["centroids"]
    return torch.tensor(centroids)
