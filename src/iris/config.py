from pathlib import Path
from typing import Any, Literal

import tomllib
import torch
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    in_path: Path
    inter_path: Path
    out_path: Path
    train_size: float
    chunk_length: int
    overlap: int


class DataloaderConfig(BaseModel):
    batch_size: int
    pin_memory: bool
    num_workers: int


class DatabaseConfig(BaseModel):
    path: Path
    num_partitions: int
    num_sub_vectors: int
    k_neighbors: int


class ModelConfig(BaseModel):
    num_queries: int
    d_model: int
    nhead: int
    hidden_dim: int
    dropout: float
    output_dim: int
    query_ini_random: bool
    device: str = "cpu"


class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float


class TrainerConfig(BaseModel):
    max_epochs: int
    checkpoint: Path
    ignore_index: int
    eval_every_n_epochs: int
    gradient_clip: float
    temperature: float
    device: str = "cpu"


class TunerConfig(BaseModel):
    path: Path
    n_trials: int
    prune: bool


class HyperparameterConfig(BaseModel):
    max_epochs: dict
    lr: dict
    weight_decay: dict
    dropout: dict
    hidden_dim: dict
    temperature: dict


class EvaluatorConfig(BaseModel):
    n_bootstraps: int
    path: Path


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


class Config(BaseModel):
    seed: int
    fast_dev_run: bool
    distributed: bool
    quantize: bool
    compile: bool
    make_dataset: bool
    make_database: bool
    tune: bool
    train: bool
    evaluate: bool
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
    evaluator: EvaluatorConfig
    task: Literal["binary", "multiclass"] = "binary"

    def model_post_init(self, __context: Any) -> None:
        device = get_device()
        self.trainer.device = device
        self.model.device = device


def get_config():
    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
    return Config(**config_data)
