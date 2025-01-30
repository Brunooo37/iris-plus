from functools import partial

import tomllib
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from ragifier.config import Config
from ragifier.data import get_dataset
from ragifier.encode import encode_sentences, get_device


def main():
    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
    cfg = Config(**config_data)

    torch.manual_seed(cfg.seed)

    model = AutoModel.from_pretrained("bert-base-uncased")
    device = get_device()
    model.to(device)

    dataset = get_dataset(device=device, cfg=cfg)
    loader = partial(DataLoader, batch_size=cfg.batch_size, pin_memory=True)
    train_loader = loader(dataset["train"])
    val_loader = loader(dataset["validation"])
    test_loader = loader(dataset["test"])

    train = encode_sentences(model, dataloader=train_loader)
    val = encode_sentences(model, dataloader=val_loader)
    test = encode_sentences(model, dataloader=test_loader)

    print(train)
    print(val)
    print(test)


if __name__ == "__main__":
    main()
