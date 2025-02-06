from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    save_path: Path,
    device: str | torch.device,
    task: Literal["binary", "multiclass"],
) -> None:
    best_val_acc: float = 0.0
    best_model_state: dict | None = None
    accuracy_metric = torchmetrics.Accuracy(task=task).to(device)
    loss_metric = torchmetrics.MeanMetric().to(device)
    for epoch in range(epochs):
        accuracy_metric.reset()
        loss_metric.reset()
        model.train()
        for id, label in train_loader:
            label = label.to(device)
            optimizer.zero_grad()
            probs = model(id, device)
            loss = loss_fn(probs, label)
            loss.backward()
            optimizer.step()
            loss_metric.update(loss.item())
            accuracy_metric.update(probs, label.long())
        train_loss = loss_metric.compute().item()
        train_acc = accuracy_metric.compute().item()
        val_loss, val_acc = evaluate_model(
            model, val_loader, loss_fn, accuracy_metric, loss_metric, device
        )
        print(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            print(f"Best model saved with Val Acc={val_acc:.4f}")
    print(f"Training complete. Best Val Accuracy: {best_val_acc:.4f}")


@torch.no_grad()
def evaluate_model(model, val_loader, loss_fn, accuracy_metric, loss_metric, device):
    model.eval()
    accuracy_metric.reset()
    loss_metric.reset()
    for batch_idx, batch_labels in val_loader:
        batch_idx = batch_idx.tolist()
        batch_labels = batch_labels.float().to(device)
        probs = model(batch_idx, device)
        loss = loss_fn(probs, batch_labels)
        loss_metric.update(loss.item())
        predictions = (probs > 0.5).long()
        accuracy_metric.update(predictions, batch_labels.long())
    val_loss = loss_metric.compute().item()
    val_acc = accuracy_metric.compute().item()
    return val_loss, val_acc
