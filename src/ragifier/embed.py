import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel


def masked_mean_pool(model_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def tensors_to_numpy(data: dict) -> dict:
    return {
        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }


def make_batch_df(batch, model):
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
    )
    batch["embedding"] = masked_mean_pool(output, mask=batch["attention_mask"])
    batch = tensors_to_numpy(batch)
    columns = ["id", "chunk_id", "text", "label", "embedding"]
    return pl.DataFrame(batch).select(columns)


# TODO: convert to iterator
@torch.no_grad()
def embed_text(model: AutoModel, dataloader: DataLoader) -> pl.DataFrame:
    dfs: list[pl.DataFrame] = []
    for batch in tqdm(dataloader):
        df = make_batch_df(batch, model)
        dfs.append(df)
    return pl.concat(dfs)
