from typing import cast

import lancedb
import polars as pl
import torch
from transformers import AutoModel

from iris.config import get_config
from iris.data import get_dataset
from iris.database import make_database
from iris.dataset import get_num_classes, make_loaders
from iris.evaluate import evaluate_model
from iris.train import train_model
from iris.tune import tune


def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)

    encoder = AutoModel.from_pretrained(cfg.model_id)
    encoder.to(cfg.model.device)
    cfg.model.d_model = encoder.config.hidden_size

    in_files = ["hyperpartisan.csv"]  # "arxiv11/data", "IMDB_Dataset.csv"
    out_files = ["hyperpartisan.parquet"]  # , "arxiv11.parquet", "imdb_dataset.parquet"
    results = []
    for in_file, out_file in zip(in_files, out_files):
        tbl_name = out_file.split(".")[0]
        if not cfg.database.path.exists() or cfg.regenerate:
            dataset = get_dataset(in_file=in_file, out_file=out_file, cfg=cfg)
            make_database(tbl_name=tbl_name, dataset=dataset, model=encoder, cfg=cfg)

        db = lancedb.connect(cfg.database.path)
        tbl = db.open_table(tbl_name)
        tbl = cast(lancedb.table.LanceTable, tbl)

        cfg.model.output_dim = get_num_classes(tbl=tbl)
        cfg.task = "binary" if cfg.model.output_dim == 2 else "multiclass"

        if cfg.tune:
            tune(cfg=cfg, tbl=tbl)
        if cfg.train:
            loaders = make_loaders(cfg=cfg, tbl=tbl)
            train_model(cfg=cfg, tbl=tbl, use_best=False)
        if cfg.evaluate:
            data = evaluate_model(
                cfg=cfg,
                dataloader=loaders.test,
                n_bootstraps=cfg.evaluator.n_bootstraps,
            )
            results.append(data)
            df = pl.DataFrame(results)
            df.write_parquet(cfg.evaluator.path)


if __name__ == "__main__":
    main()
