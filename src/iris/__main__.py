from typing import cast

import lancedb
import polars as pl
import torch
from transformers import AutoModel  # , BitsAndBytesConfig

from iris.config import get_config
from iris.data import make_dataset
from iris.database import make_database
from iris.dataset import get_num_classes, make_loaders
from iris.evaluate import evaluate_model
from iris.train import train_model
from iris.tune import tune


def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    encoder = AutoModel.from_pretrained(
        cfg.model_id,
        # quantization_config=quant_config,
        # torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    encoder.to(cfg.model.device)
    cfg.model.d_model = encoder.config.hidden_size

    in_file = "arxiv11/data"
    tbl_name = "arxiv11"

    if cfg.make_dataset:
        make_dataset(in_file=in_file, cfg=cfg)

    if not cfg.database.path.exists() or cfg.make_database:
        make_database(tbl_name=tbl_name, model=encoder, cfg=cfg)

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
        data = evaluate_model(cfg=cfg, dataloader=loaders.test)
        df = pl.DataFrame(data)
        df.write_parquet(cfg.evaluator.path)


if __name__ == "__main__":
    main()
