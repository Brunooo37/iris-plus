from functools import partial
from pathlib import Path

import polars as pl
import polars.selectors as cs
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

from ragifier.config import Config


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
        .with_columns(chunk_id=pl.int_ranges(pl.col("text").list.len()))
        .explode("text", "chunk_id")
        .select("id", "chunk_id", "text", "label")
    )


def chunk_text(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    texts = []
    for text in df["text"]:
        chunks = make_chunks(text, chunk_length=cfg.chunk_length, overlap=cfg.overlap)
        texts.append(chunks)
    df = format_dataframe(df, texts)
    return df


def make_arxiv11_dataset(path: Path) -> pl.DataFrame:
    rows = [
        {"text": path.read_text(), "label": path.parent.name}
        for path in path.rglob("*.txt")
    ]
    df = pl.LazyFrame(rows)
    df = (
        df.with_columns(pl.col("label").rank("dense"))
        .with_row_index(name="id")
        .with_columns(cs.integer().cast(pl.Int32))
        .collect()
    )
    return df


def tokenize(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=max_length
    )


def train_val_test_split(dataset: Dataset, cfg: Config) -> Dataset:
    dataset = DatasetDict({"train": dataset})
    train_testvalid = dataset["train"].train_test_split(
        test_size=cfg.train_size, shuffle=False
    )
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5, shuffle=False)
    return DatasetDict(
        {
            "train": train_testvalid["train"],
            "test": test_valid["test"],
            "validation": test_valid["train"],
        }
    )


# TODO add other datasets
def get_df_fn(path: Path):
    if path.name == "arxiv11":
        fn = make_arxiv11_dataset
    # elif path.name == "hyperpartisan":
    #     fn = make_hyperpartisan_dataset
    else:
        raise ValueError(f"Unknown dataset: {path.name}")
    return partial(fn, path=path)


def make_dataset(cfg: Config) -> DatasetDict:
    make_df = get_df_fn(path=cfg.output_path)
    df: pl.DataFrame = make_df(path=cfg.input_path)
    if cfg.fast_dev_run:
        df = df.head(10)
    df = chunk_text(df=df, cfg=cfg)
    dataset = Dataset.from_polars(df)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenize_fn = partial(tokenize, tokenizer=tokenizer, max_length=cfg.max_length)
    dataset = dataset.map(tokenize_fn, batched=True, desc="Tokenizing")
    dataset = train_val_test_split(dataset, cfg=cfg)
    return dataset


def get_dataset(device: str, cfg: Config) -> DatasetDict:
    if cfg.output_path.exists():
        dataset = load_from_disk(cfg.output_path)
    else:
        dataset = make_dataset(cfg=cfg)
        dataset.save_to_disk(cfg.output_path)
    return dataset.with_format("torch", device=device)
