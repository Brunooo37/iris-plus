import lancedb
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import AutoModelForTextEncoding

from rassifier.config import get_config
from rassifier.data import get_dataset
from rassifier.database import make_database
from rassifier.dataset import get_ini_queries, make_loaders
from rassifier.evaluate import evaluate_model
from rassifier.model import get_num_classes
from rassifier.train import train_model
from rassifier.tune import tune


def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)

    dataset = get_dataset(cfg=cfg)
    dataloader = DataLoader(dataset, **cfg.dataloader.model_dump())  # type: ignore

    encoder = AutoModelForTextEncoding.from_pretrained(cfg.model_id)
    encoder.to(cfg.trainer.device)

    if not cfg.database.path.exists() or cfg.regenerate:
        make_database(model=encoder, dataloader=dataloader, cfg=cfg.database)

    db = lancedb.connect(cfg.database.path)
    tbl = db.open_table(cfg.database.tbl_name)

    cfg.model.d_model = encoder.config.hidden_size
    cfg.model.output_dim = get_num_classes(tbl=tbl)

    ini_queries = get_ini_queries(tbl=tbl, cfg=cfg)
    loaders = make_loaders(cfg=cfg, tbl=tbl, ini_queries=ini_queries)

    if cfg.tune:
        tune(cfg=cfg, loaders=loaders, tbl=tbl, ini_queries=ini_queries)

    if cfg.train:
        train_model(
            cfg=cfg,
            loaders=loaders,
            ini_queries=ini_queries,
            use_best=True,
            validate=False,
        )

    if cfg.evaluate:
        task = "binary" if cfg.model.output_dim == 2 else "multiclass"
        metrics = [Accuracy(task=task, num_classes=cfg.model.output_dim)]
        results = evaluate_model(cfg=cfg, dataloader=loaders.test, metrics=metrics)
        print(results["mean"], results["std"])


if __name__ == "__main__":
    main()
