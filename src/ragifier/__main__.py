import tomllib
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from ragifier.config import Config
from ragifier.data import get_dataset
from ragifier.encode import encode_text


def main():
    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
    cfg = Config(**config_data)

    torch.manual_seed(cfg.seed)

    dataset = get_dataset(cfg=cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=True)  # type: ignore

    model = AutoModel.from_pretrained(cfg.model)
    model.to(cfg.device)
    results = encode_text(model, dataloader=dataloader)
    print(results)


if __name__ == "__main__":
    main()
