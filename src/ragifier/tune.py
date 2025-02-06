import json

import optuna
import torch
import torch.nn as nn
from optuna import Trial
from optuna.pruners import HyperbandPruner
from optuna.samplers import QMCSampler, TPESampler

from ragifier.config import Config
from ragifier.dataset import DataLoaders
from ragifier.train import make_trainer, set_hparams


class Objective:
    def __init__(self, dataloaders: DataLoaders, cfg: Config):
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.best_validation_loss = float("inf")

    def sample_hyperparameters(self, trial: optuna.Trial):
        return {
            "lr": trial.suggest_float(**self.cfg.hparams.lr),
            "weight_decay": trial.suggest_float(**self.cfg.hparams.weight_decay),
        }

    def __call__(self, trial: Trial) -> float:
        hyperparams = self.sample_hyperparameters(trial=trial)
        cfg = set_hparams(cfg=self.cfg, **hyperparams)
        trainer = make_trainer(cfg=cfg, dataloaders=self.dataloaders)
        trainer.train(validate=True)
        if trainer.validation_loss < self.best_validation_loss:
            self.best_validation_loss = trainer.validation_loss
            with open(self.cfg.tuner.path, "w") as f:
                json.dump(hyperparams, f)
            torch.save(trainer.model.state_dict(), cfg.tuner.checkpoint)
        return trainer.validation_loss


def make_study(cfg: Config, sampler: optuna.samplers.BaseSampler) -> optuna.study.Study:
    if cfg.tuner.prune:
        pruner = HyperbandPruner(
            min_resource=cfg.trainer.max_epochs // 4,
            max_resource=cfg.trainer.max_epochs,
        )
    else:
        pruner = None
    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )
    return optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name="tune",
    )


def tune_hyperparameters(dataloaders: DataLoaders, cfg: Config):
    objective = Objective(cfg=cfg, dataloaders=dataloaders)
    half_trials = cfg.tuner.n_trials // 2
    sampler = QMCSampler(seed=cfg.seed)
    study = make_study(cfg=cfg, sampler=sampler)
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=cfg.seed)
    study.optimize(func=objective, n_trials=half_trials)
    if cfg.verbose:
        print("Best model hyperparameters:")
        print(json.dumps(study.best_params, indent=4))
        print(f"Best model checkpoint saved to: {cfg.tuner.checkpoint}")
        print(f"Best model hyperparameters saved to: {cfg.tuner.path}")


def load_best_checkpoint(cfg: Config, model_class: type[nn.Module]) -> nn.Module:
    model_weights = torch.load(cfg.tuner.checkpoint, weights_only=True)
    with open(cfg.tuner.path, "r") as f:
        hyperparams = json.load(f)
    cfg = set_hparams(cfg=cfg, **hyperparams)
    model: nn.Module = model_class(**cfg.model.model_dump())
    model.load_state_dict(model_weights)
    model.to(cfg.trainer.device)
    return model
