import json

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ragifier.config import Config, TrainerConfig
from ragifier.dataset import DataLoaders
from ragifier.model import Ragifier


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        dataloaders: DataLoaders,
        cfg: TrainerConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.progress_bar = tqdm(range(cfg.max_epochs), desc="Epoch")
        self.cfg = cfg
        self.train_loss = float("inf")
        self.validation_loss = float("inf")

    def train_step(
        self, inputs: torch.Tensor, labels: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        self.optimizer.zero_grad()
        inputs = inputs.to(self.cfg.device)
        labels = labels.to(self.cfg.device)
        outputs = self.model(inputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.gradient_clip)
        self.optimizer.step()
        return loss

    def train(self, validate: bool = True) -> None:
        self.model.to(self.cfg.device)
        for epoch in self.progress_bar:
            self.model.train()
            loss = 0
            for inputs, labels, padding_mask in self.dataloaders.train:
                loss += self.train_step(inputs, labels, padding_mask)
            loss /= len(self.dataloaders.train)
            self.train_loss = loss
            postfix = f"Train loss: {loss:.4f}"
            if epoch % self.cfg.eval_every_n_epochs == 0:
                if validate:
                    self.validate()
                    postfix += f", Val loss: {self.validation_loss:.4f}"
            self.progress_bar.set_postfix_str(postfix)

    @torch.no_grad()
    def validate(self) -> None:
        self.model.eval()
        loss = 0
        for inputs, labels, padding_mask in self.dataloaders.validation:
            inputs = inputs.to(self.cfg.device)
            labels = labels.to(self.cfg.device)
            outputs = self.model(inputs, padding_mask)
            loss += self.criterion(outputs, labels)
        loss /= len(self.dataloaders.validation)
        self.validation_loss = loss


@torch.no_grad()
def predict(
    model: nn.Module, dataloader: DataLoader, device: str | torch.device
) -> list[torch.Tensor]:
    model.eval()
    predictions = []
    for inputs, labels, padding_mask in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs, padding_mask)
        predictions.append(outputs)
    return predictions


def make_trainer(cfg: Config, dataloaders: DataLoaders) -> Trainer:
    model = Ragifier(**cfg.model.model_dump())
    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
    criterion = nn.CrossEntropyLoss()
    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloaders=dataloaders,
        cfg=cfg.trainer,
    )


def set_hparams(cfg: Config, lr: float, weight_decay: float) -> Config:
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = weight_decay
    return cfg


def train_model(
    dataloaders: DataLoaders, cfg: Config, use_best: bool = False, validate: bool = True
):
    if use_best:
        hyperparameters = json.load(open(cfg.tuner.path, "r"))
        cfg = set_hparams(cfg=cfg, **hyperparameters)
    trainer = make_trainer(cfg=cfg, dataloaders=dataloaders)
    trainer.train(validate=validate)
    torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
