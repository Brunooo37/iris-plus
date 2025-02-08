import json

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from rassifier.config import Config, TrainerConfig
from rassifier.dataset import DataLoaders
from rassifier.loss import query_penalty
from rassifier.model import Rassifier, make_model


class Trainer:
    def __init__(
        self,
        model: Rassifier,
        optimizer: Optimizer,
        criterion: nn.Module,
        loaders: DataLoaders,
        cfg: TrainerConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loaders = loaders
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
        padding_mask = padding_mask.to(self.cfg.device)
        outputs = self.model(inputs, padding_mask)
        loss: torch.Tensor = self.criterion(outputs, labels)
        loss += query_penalty(self.model.queries, self.cfg.temperature)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.gradient_clip)
        self.optimizer.step()
        return loss

    def train(self, validate: bool = True) -> None:
        self.model.to(self.cfg.device)
        for epoch in self.progress_bar:
            self.model.train()
            loss = 0
            for inputs, labels, padding_mask in self.loaders.train:
                loss += self.train_step(inputs, labels, padding_mask)

                train_queries = self.model.queries.detach().cpu().numpy()
                self.loaders.train.dataset.queries = train_queries  # type: ignore
            loss /= len(self.loaders.train)
            self.train_loss = loss
            postfix = f"Train loss: {loss:.4f}"
            val_queries = self.model.queries.detach().cpu().numpy()
            self.loaders.validation.dataset.queries = val_queries  # type: ignore
            if epoch % self.cfg.eval_every_n_epochs == 0:
                if validate:
                    self.validate()
                    postfix += f", Val loss: {self.validation_loss:.4f}"
            self.progress_bar.set_postfix_str(postfix)
        test_queries = self.model.queries.detach().cpu().numpy()
        self.loaders.test.dataset.queries = test_queries  # type: ignore

    @torch.no_grad()
    def validate(self) -> None:
        self.model.eval()
        loss = 0
        for inputs, labels, padding_mask in self.loaders.validation:
            inputs = inputs.to(self.cfg.device)
            labels = labels.to(self.cfg.device)
            padding_mask = padding_mask.to(self.cfg.device)
            outputs = self.model(inputs, padding_mask)
            loss += self.criterion(outputs, labels)
        loss /= len(self.loaders.validation)
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
        padding_mask = padding_mask.to(device)
        outputs = model(inputs, padding_mask)
        predictions.append(outputs)
    return predictions


def make_trainer(cfg: Config, loaders: DataLoaders) -> Trainer:
    ini_queries = loaders.train.dataset.queries  # type: ignore
    model = make_model(ini_queries=ini_queries, cfg=cfg.model)
    model.to(cfg.trainer.device)
    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump(), fused=True)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.trainer.ignore_index)
    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        loaders=loaders,
        cfg=cfg.trainer,
    )


def set_hyperparams(
    cfg: Config, max_epochs: int, temperature: float, lr: float, weight_decay: float
) -> Config:
    cfg.trainer.max_epochs = max_epochs
    cfg.trainer.temperature = temperature
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = weight_decay

    return cfg


def train_model(
    cfg: Config,
    loaders: DataLoaders,
    use_best: bool = False,
    validate: bool = True,
):
    if use_best:
        hyperparameters = json.load(open(cfg.tuner.path, "r"))
        cfg = set_hyperparams(cfg=cfg, **hyperparameters)
    trainer = make_trainer(cfg=cfg, loaders=loaders)
    trainer.train(validate=validate)
    torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
