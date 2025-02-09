import polars as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.wrappers import BootStrapper

from rassifier.config import Config


def bootstrap_metric(
    metric, outputs: torch.Tensor, labels: torch.Tensor, n_bootstraps: int
):
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=True, std=True, raw=True
    )
    bootstrap.update(outputs, labels)
    return bootstrap.compute()


def bootstrap_metrics(cfg: Config, metrics: dict[str, Metric], n_bootstraps: int):
    bootstrapped = {}
    for name, metric in metrics.items():
        bootstrapper = BootStrapper(
            metric, num_bootstraps=n_bootstraps, mean=True, std=True, raw=True
        )
        bootstrapper.to(cfg.trainer.device)
        bootstrapped[name] = bootstrapper
    return bootstrapped


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    cfg: Config,
    dataloader: DataLoader,
    metrics: dict[str, Metric],
    n_bootstraps: int = 1000,
) -> dict:
    results = {}
    metrics = bootstrap_metrics(cfg, metrics, n_bootstraps)
    for name, metric in metrics.items():
        for inputs, labels, padding_mask in dataloader:
            inputs = inputs.to(cfg.trainer.device)
            labels = labels.to(cfg.trainer.device)
            padding_mask = padding_mask.to(cfg.trainer.device)
            outputs = model(inputs, padding_mask)
            pred = outputs.argmax(dim=1)
            metric.update(pred, labels)
        results[name] = metric.compute()
        metric.reset()
    return results


# TODO generate metrics for subsets of the data
def evaluate_subsets(df: pl.DataFrame, groups: list[str]):
    pass
