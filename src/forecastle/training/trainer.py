from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch.utils.data import DataLoader

    from forecastle.config import TrainingConfig


@dataclass(frozen=True)
class FitResult:
    best_val_loss: float
    epochs_ran: int
    training_time_seconds: float
    checkpoint_path: Path


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_path: Path,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        after_validation: Callable[[int, float], None] | None = None,
    ) -> FitResult:
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        start = time.perf_counter()
        epochs_ran = 0

        for epoch in range(1, self.config.epochs + 1):
            epochs_ran = epoch
            self._train_epoch(train_loader)
            val_loss = self.evaluate_loss(val_loader)
            if after_validation is not None:
                after_validation(epoch, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint()
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.config.patience:
                break

        training_time = time.perf_counter() - start
        self._load_checkpoint()
        return FitResult(
            best_val_loss=best_val_loss,
            epochs_ran=epochs_ran,
            training_time_seconds=training_time,
            checkpoint_path=self.checkpoint_path,
        )

    def evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                predictions = self.model(features)
                losses.append(float(self.loss_fn(predictions, targets).item()))
        return sum(losses) / max(len(losses), 1)

    def predict(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, float]:
        self.model.eval()
        predictions = []
        actuals = []
        start = time.perf_counter()
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                outputs = self.model(features).detach().cpu()
                predictions.append(outputs)
                actuals.append(targets.detach().cpu())
        inference_time = time.perf_counter() - start
        return torch.cat(actuals), torch.cat(predictions), inference_time

    def _train_epoch(self, loader: DataLoader) -> None:
        self.model.train()
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(features)
            loss = self.loss_fn(predictions, targets)
            loss.backward()
            self.optimizer.step()

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def _load_checkpoint(self) -> None:
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
