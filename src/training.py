"""Training helpers for NeuralCORA models."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import CoraDataset

__all__ = ["fit", "create_predictions"]


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int,
    device: torch.device | str,
    valid_loader: DataLoader | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, list[float]]:
    """Train ``model`` for ``epochs`` epochs and return loss history."""

    device = torch.device(device)
    model.to(device)

    if log_fn is None:
        log_fn = print

    history = {"train": [], "valid": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        history["train"].append(train_loss)

        val_loss = float("nan")
        if valid_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(valid_loader.dataset)
        history["valid"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")
        history["lr"].append(current_lr)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitor = val_loss if not np.isnan(val_loss) else train_loss
                scheduler.step(monitor)
            else:
                scheduler.step()

        msg = f"Epoch {epoch + 1}/{epochs}: train {train_loss:.4f}"
        if not np.isnan(val_loss):
            msg += f" val {val_loss:.4f}"
        log_fn(msg)

    return history


def create_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: CoraDataset,
    device: torch.device | str,
) -> xr.Dataset:
    """Run model inference over ``loader`` and return an ``xarray.Dataset``."""

    import xarray as xr  # Local import to avoid hard dependency when unused.

    model.eval()
    device = torch.device(device)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            outputs = model(xb).cpu().numpy()
            preds.append(outputs)
    predictions = np.concatenate(preds, axis=0)
    predictions = dataset.unnormalize(predictions)
    return dataset.to_xarray(predictions, dataset.valid_time)
