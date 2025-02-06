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
    path: Path
    tbl_name: str
    num_partitions: int
    num_sub_vectors: int


class ModelConfig(BaseModel):
    num_queries: int
    k_neighbors: int
    nhead: int
    dim_feedforward: int
    dropout: float
    num_layers: int
    temperature: float
    query_ini_random: bool


class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float


class TrainerConfig(BaseModel):
    max_epochs: int
    save_path: Path
    ignore_index: int
    eval_every_n_epochs: int
    gradient_clip: float
    device = "cpu"


class TunerConfig(BaseModel):
    path: Path
    checkpoint: Path
    n_trials: int
    prune: bool


class HyperparameterConfig(BaseModel):
    lr: dict
    weight_decay: dict


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
    model_id: str
    verbose: bool
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    database: DatabaseConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    tuner: TunerConfig
    hparams: HyperparameterConfig

    def model_post_init(self, __context: Any) -> None:
        self.device = get_device()


def get_config():
    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
    return Config(**config_data)
