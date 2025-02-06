from functools import partial
from pathlib import Path
from typing import cast

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
    df = (
        df.with_columns(pl.col("label").rank("dense") - 1)
        .with_row_index(name="id")
        .with_columns(cs.integer().cast(pl.Int32))
        .collect()
    )
    return df


def make_hyperpartisan_dataset(path: Path):
    df = pl.read_csv(path)
    df = df.with_columns(label=pl.col("Hyperpartisan").cast(pl.Int32))
    df = df.rename({"Article ID": "id", "Content": "text"})
    df = df.select("id", "label", "text")
    return df


# TODO add other datasets
def make_df(in_path: Path, name: str) -> pl.DataFrame:
    if name == "arxiv11":
        return make_arxiv11_dataset(path=in_path)
    elif name == "hyperpartisan.parquet":
        return make_hyperpartisan_dataset(path=in_path)
    else:
        raise ValueError(f"Unknown dataset: {name}")


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


def make_dataset(cfg: Config) -> Dataset:
    df = make_df(in_path=cfg.dataset.input_path, name=cfg.dataset.output_path.name)
    if cfg.fast_dev_run:
        df = df.head(10)
    df = chunk_text(df=df, cfg=cfg.dataset)
    dataset = Dataset.from_polars(df)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenize_fn = partial(
        tokenize, tokenizer=tokenizer, max_length=cfg.dataset.max_length
    )
    dataset = dataset.map(tokenize_fn, batched=True, desc="Tokenizing")
    return dataset


def get_dataset(cfg: Config) -> Dataset:
    if not cfg.dataset.output_path.exists() or cfg.regenerate:
        dataset = make_dataset(cfg=cfg)
        dataset.save_to_disk(cfg.dataset.output_path)
    else:
        dataset = load_from_disk(cfg.dataset.output_path)
        dataset = cast(Dataset, dataset)
    return dataset.with_format("torch", device=cfg.trainer.device)
