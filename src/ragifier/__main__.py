import lancedb
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForTextEncoding

from ragifier.config import get_config
from ragifier.data import get_dataset, get_task
from ragifier.database import make_database
from ragifier.dataset import get_initial_queries, make_dataloaders
from ragifier.model import make_model
from ragifier.train import train_model


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
    initial_queries = get_initial_queries(tbl=tbl, vector_dim=vector_dim, cfg=cfg)
    train, val, test = make_dataloaders(
        tbl=tbl, vector_dim=vector_dim, initial_queries=initial_queries, cfg=cfg
    )  # TODO: test is unused
    model = make_model(
        tbl=tbl, vector_dim=vector_dim, initial_queries=initial_queries, cfg=cfg
    )
    task = get_task(cfg.dataset.output_path.stem)
    loss_fn = nn.BCEWithLogitsLoss() if task == "binary" else nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
    train_model(
        model=model,
        train_loader=train,
        val_loader=val,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.trainer.epochs,
        save_path=cfg.trainer.save_path,
        device=cfg.device,
        task=task,
    )


if __name__ == "__main__":
    main()
