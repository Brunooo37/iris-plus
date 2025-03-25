from functools import partial
from typing import cast

import lancedb
import polars as pl
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, BitsAndBytesConfig

from iris.config import Config


def masked_mean_pool(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = output[0]
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def tensors_to_numpy(data: dict) -> dict:
    d = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.cpu().numpy()
        else:
            d[k] = v
    return d


def make_batch_df(batch, model):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    batch["vector"] = masked_mean_pool(output, batch["attention_mask"])
    batch = tensors_to_numpy(batch)
    columns = ["id", "chunk_id", "offset", "label", "vector"]
    df = pl.DataFrame(batch).select(columns)
    return df


@torch.no_grad()
def make_batches(model: AutoModel, dataloader: DataLoader):
    for batch in tqdm(dataloader):
        df = make_batch_df(batch, model)
        yield df


def make_encoder(cfg: Config):
    if cfg.quantize:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        encoder = AutoModel.from_pretrained(
            cfg.model_id,
            quantization_config=quant_config,
            torch_dtype="auto",
            device_map="auto",
            tp_plan="auto",
        )
    else:
        encoder = AutoModel.from_pretrained(
            cfg.model_id, torch_dtype="auto", device_map="auto"
        )
    if cfg.compile:
        encoder.forward = torch.compile(
            encoder.forward, mode="reduce-overhead", fullgraph=True
        )
    encoder.to(cfg.model.device)
    return encoder


def make_database(tbl_name: str, cfg: Config):
    # type: ignore
    dataset = load_from_disk(cfg.dataset.out_path)
    dataset = dataset.with_format("torch", device=cfg.trainer.device)
    dataset = cast(torch.utils.data.Dataset, dataset)
    dataloader = DataLoader(dataset, **cfg.dataloader.model_dump())
    db = lancedb.connect(cfg.database.path)
    model = make_encoder(cfg=cfg)
    model.eval()
    batch_fn = partial(make_batches, model=model, dataloader=dataloader)
    tbl = db.create_table(tbl_name, data=batch_fn(), mode="overwrite")
    tbl.create_scalar_index("id", index_type="BTREE")
    tbl.create_scalar_index("chunk_id", index_type="BITMAP")
    tbl.create_index(
        num_partitions=cfg.database.num_partitions,
        num_sub_vectors=cfg.database.num_sub_vectors,
    )


def get_centroids(lance_tbl):
    vector_index = lance_tbl.index_statistics("vector_idx")
    centroids = vector_index["indices"][0]["centroids"]
    return torch.tensor(centroids)
