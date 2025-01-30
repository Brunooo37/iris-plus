from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    fast_dev_run: bool
    seed: int
    input_path: Path
    output_path: Path
    train_size: float
    max_length: int
    batch_size: int
    chunk_length: int
    overlap: int
