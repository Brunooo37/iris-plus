from pathlib import Path
from typing import Any

import tomllib
import torch
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    input_path: Path
    output_path: Path
    train_size: float
    max_length: int
    chunk_length: int
    overlap: int


class DataloaderConfig(BaseModel):
    batch_size: int
    pin_memory: bool
    num_workers: int


class DatabaseConfig(BaseModel):
    database_path: Path
    table_name: str


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


class Config(BaseModel):
    fast_dev_run: bool
    regenerate: bool
    seed: int
    model: str
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    database: DatabaseConfig
    device: str | None = None

    def model_post_init(self, __context: Any) -> None:
        self.device = get_device()


def get_config():
    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
    return Config(**config_data)
