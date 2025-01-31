import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from ragifier.config import get_config
from ragifier.data import get_dataset
from ragifier.encode import encode_text


def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)

    dataset = get_dataset(cfg=cfg)
    dataloader = DataLoader(dataset, **cfg.dataloader.model_dump())  # type: ignore

    model = AutoModel.from_pretrained(cfg.model)
    model.to(cfg.device)
    results = encode_text(model, dataloader=dataloader)
    print(results)


if __name__ == "__main__":
    main()
