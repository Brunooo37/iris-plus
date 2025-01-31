from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


class DatasetConfig(BaseModel):
    input_path: Path
    output_path: Path
    train_size: float
    max_length: int
    chunk_length: int
    overlap: int


class DatabaseConfig(BaseModel):
    database_path: Path
    table_name: str


class Config(BaseModel):
    fast_dev_run: bool
    regenerate: bool
    seed: int
    model: str
    batch_size: int
    dataset: DatasetConfig
    database: DatabaseConfig
    device: str | None = None

    def model_post_init(self, __context: Any) -> None:
        self.device = get_device()
