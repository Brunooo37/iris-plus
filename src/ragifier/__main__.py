import duckdb
import lancedb
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForTextEncoding

from ragifier.config import get_config
from ragifier.data import get_dataset
from ragifier.database import make_database
from ragifier.dataset import TableDataset, get_initial_queries
from ragifier.model import make_model
from ragifier.train import train_model


def get_task(dataset: str):
    if dataset == "arxiv11":
        return "multiclass"
    elif dataset == "hyperpartisan.parquet":
        return "binary"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)

    dataset = get_dataset(cfg=cfg)
    dataloader = DataLoader(dataset, **cfg.dataloader.model_dump())  # type: ignore

    model = AutoModelForTextEncoding.from_pretrained(cfg.model_id)
    model.to(cfg.device)

    if not cfg.database.path.exists() or cfg.regenerate:
        make_database(model=model, dataloader=dataloader, cfg=cfg.database)

    db = lancedb.connect(cfg.database.path)
    tbl = db.open_table(cfg.database.tbl_name)

    vector_dim = tbl.schema.field("vector").type.list_size

    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())

    _ = tbl.to_lance()  # type: ignore
    df = pl.DataFrame(duckdb.sql("SELECT DISTINCT id, label FROM _").to_df())

    train, val = train_test_split(df, test_size=0.2, random_state=cfg.seed)
    # val, test = train_test_split(temp, test_size=0.5, random_state=cfg.seed)

    initial_queries = get_initial_queries(tbl=tbl, vector_dim=vector_dim, cfg=cfg)

    dataset = TableDataset(
        tbl=tbl,
        vector_dim=vector_dim,
        k_neighbours=cfg.model.k_neighbors,
        ignore_index=cfg.ignore_index,
        initial_queries=initial_queries.numpy(),
    )

    train_loader = DataLoader(train, **cfg.dataloader.model_dump(), shuffle=True)
    val_loader = DataLoader(val, **cfg.dataloader.model_dump())
    # test_loader = DataLoader(test, **cfg.dataloader.model_dump())

    model = make_model(
        tbl=tbl,
        vector_dim=vector_dim,
        cfg=cfg,
        initial_queries=initial_queries,
    )
    task = get_task(cfg.dataset.output_path.stem)
    loss_fn = nn.BCEWithLogitsLoss() if task == "binary" else nn.CrossEntropyLoss()

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.trainer.epochs,
        save_path=cfg.trainer.save_path,
        device=cfg.device,
        task=task,
    )


if __name__ == "__main__":
    main()
