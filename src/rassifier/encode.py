import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

# def masked_mean_pool(model_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#     token_embeddings = model_output[0]
#     input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#         input_mask_expanded.sum(1), min=1e-9
#     )


def tensors_to_numpy(data: dict) -> dict:
    d = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.cpu().numpy()
        else:
            d[k] = v
    return d


def make_batch_df(batch, model):
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
    )
    batch["vector"] = output["pooler_output"]
    batch = tensors_to_numpy(batch)
    columns = ["id", "text", "label", "vector"]
    df = pl.DataFrame(batch).select(columns)
    return df


@torch.no_grad()
def make_batches(model: AutoModel, dataloader: DataLoader):
    for batch in tqdm(dataloader):
        df = make_batch_df(batch, model)
        yield df
