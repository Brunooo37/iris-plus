from functools import partial
from pathlib import Path
from typing import Callable, cast

import polars as pl
import polars.selectors as cs
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from rassifier.config import Config, DatasetConfig


def make_arxiv11_dataset(path: Path) -> pl.DataFrame:
    rows = [
        {"text": path.read_text(), "label": path.parent.name}
        for path in path.rglob("*.txt")
    ]
    df = pl.LazyFrame(rows)
    include = ["cs.AI", "cs.DS", "cs.PL"]  # "cs.IT" is missing
    df = (
        df.filter(pl.col("label").is_in(include))
        .with_columns((pl.col("label").rank("dense") - 1).cast(pl.Int32))
        .with_row_index(name="id")
        .with_columns(cs.integer().cast(pl.Int32))
        .collect()
    )
    return df


def make_hyperpartisan_dataset(path: Path):
    df = pl.read_csv(path)
    df = df.with_columns(label=pl.col("Hyperpartisan").cast(pl.Int32))
    df = df.rename({"Article ID": "id", "Content": "text"})
    return df


def make_imdb_dataset(path: Path):
    df = (
        pl.read_csv(path)
        .with_row_index("id")
        .rename({"review": "text", "sentiment": "label"})
        .with_columns(pl.col("label").replace({"positive": 1, "negative": 0}))
    )
    return df


def get_make_df_fn(name: str) -> Callable[[Path], pl.DataFrame]:
    match name:
        case "arxiv11/data":
            return make_arxiv11_dataset
        case "hyperpartisan.csv":
            return make_hyperpartisan_dataset
        case "IMDB_Dataset.csv":
            return make_imdb_dataset
        case _:
            raise ValueError(f"Unknown dataset: {name}")


# FIXME need to chunk tokens, not text
def make_chunks(text, chunk_length, overlap) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_length - overlap):
        chunk = " ".join(words[i : i + chunk_length])
        chunks.append(chunk)
    return chunks


def format_dataframe(df: pl.DataFrame, texts: list[str]) -> pl.DataFrame:
    return (
        df.with_columns(text=pl.Series(texts))
        .explode("text")
        .select("id", "text", "label")
    )


# FIXME need to chunk tokens, not text
def chunk_text(df: pl.DataFrame, cfg: DatasetConfig) -> pl.DataFrame:
    texts = []
    for text in df["text"]:
        chunks = make_chunks(text, chunk_length=cfg.chunk_length, overlap=cfg.overlap)
        texts.append(chunks)
    df = format_dataframe(df, texts)
    return df


def tokenize(batch, tokenizer, max_length) -> AutoTokenizer:
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=max_length
    )


def make_dataset(df: pl.DataFrame, cfg: Config) -> Dataset:
    if cfg.fast_dev_run:
        df = df.head(10)
    # FIXME need to chunk tokens, not text
    df = chunk_text(df=df, cfg=cfg.dataset)
    dataset = Dataset.from_polars(df)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenize_fn = partial(
        tokenize, tokenizer=tokenizer, max_length=cfg.dataset.max_length
    )
    dataset = dataset.map(tokenize_fn, batched=True, desc="Tokenizing")
    return dataset


def get_dataset(in_file: str, out_file: str, cfg: Config) -> Dataset:
    if not cfg.dataset.out_path.exists() or cfg.regenerate:
        make_df = get_make_df_fn(name=in_file)
        df = make_df(cfg.dataset.in_path / in_file).select("id", "label", "text")
        df.write_parquet(cfg.dataset.inter_path / out_file)
        dataset = make_dataset(df=df, cfg=cfg)
        dataset.save_to_disk(cfg.dataset.out_path)
    else:
        dataset = load_from_disk(cfg.dataset.out_path)
        dataset = cast(Dataset, dataset)
    return dataset.with_format("torch", device=cfg.trainer.device)
