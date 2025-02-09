from typing import cast

import lancedb
import torch
from torchmetrics import Accuracy
from transformers import AutoModelForTextEncoding

from rassifier.config import get_config
from rassifier.database import make_database
from rassifier.dataset import get_num_classes, make_loaders
from rassifier.evaluate import evaluate_model
from rassifier.model import Rassifier
from rassifier.train import train_model
from rassifier.tune import load_best_checkpoint, tune


def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)

    encoder = AutoModelForTextEncoding.from_pretrained(cfg.model_id)
    encoder.to(cfg.model.device)
    cfg.model.d_model = encoder.config.hidden_size

    if not cfg.database.path.exists() or cfg.regenerate:
        make_database(model=encoder, cfg=cfg)

    db = lancedb.connect(cfg.database.path)
    tbl = db.open_table(cfg.database.tbl_name)
    tbl = cast(lancedb.table.LanceTable, tbl)

    cfg.model.output_dim = get_num_classes(tbl=tbl)
    cfg.task = "binary" if cfg.model.output_dim == 2 else "multiclass"

    if cfg.tune:
        tune(cfg=cfg, tbl=tbl)

    if cfg.train:
        loaders = make_loaders(cfg=cfg, tbl=tbl)
        train_model(cfg=cfg, tbl=tbl, use_best=False)

    if cfg.evaluate:
        metrics = {
            "accuracy": Accuracy(
                task=cfg.task,
                num_classes=cfg.model.output_dim,
                ignore_index=cfg.trainer.ignore_index,
            )
        }
        model = load_best_checkpoint(cfg=cfg, model_class=Rassifier)
        results = evaluate_model(
            model,
            cfg=cfg,
            dataloader=loaders.test,
            metrics=metrics,
            n_bootstraps=cfg.evaluator.n_bootstraps,
        )
        accuracy = results["accuracy"]
        print(accuracy["mean"].item(), accuracy["std"].item())


if __name__ == "__main__":
    main()
