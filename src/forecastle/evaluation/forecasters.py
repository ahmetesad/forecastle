from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from forecastle.evaluation.types import FitSummary
from forecastle.models import build_model
from forecastle.training import Trainer
from forecastle.utils.seed import seed_everything

if TYPE_CHECKING:
    from pathlib import Path

    from forecastle.config import ModelRunConfig, TrainingConfig
    from forecastle.data import DataModule


class FittedForecaster(Protocol):
    name: str
    summary: FitSummary

    def predict(self, raw_window: np.ndarray, previous_price: float) -> float: ...


class PersistenceForecaster:
    def __init__(self, target_transform: str, fold: int) -> None:
        self.name = "naive_persistence"
        self.target_transform = target_transform
        self.summary = FitSummary(model=self.name, fold=fold)

    def predict(self, raw_window: np.ndarray, previous_price: float) -> float:
        del raw_window
        if self.target_transform == "price":
            return previous_price
        return 0.0


class LinearForecaster:
    def __init__(self, datamodule: DataModule, fold: int) -> None:
        self.name = "linear_regression"
        self.datamodule = datamodule
        train_x, train_y = dataset_to_numpy(datamodule.train_dataset)
        start = time.perf_counter()
        self.regressor = LinearRegression()
        self.regressor.fit(flatten_windows(train_x), train_y.reshape(-1))
        training_time = time.perf_counter() - start
        self.summary = FitSummary(
            model=self.name,
            fold=fold,
            training_time_seconds=training_time,
        )

    def predict(self, raw_window: np.ndarray, previous_price: float) -> float:
        del previous_price
        scaled_window = scale_window(raw_window, self.datamodule)
        start = time.perf_counter()
        predicted_scaled = float(self.regressor.predict(flatten_windows(scaled_window[None]))[0])
        self.summary.inference_time_seconds += time.perf_counter() - start
        return predicted_scaled * self.datamodule.target_std + self.datamodule.target_mean


class NeuralForecaster:
    def __init__(
        self,
        model_config: ModelRunConfig,
        training_config: TrainingConfig,
        datamodule: DataModule,
        device: torch.device,
        checkpoint_path: Path,
        fold: int,
    ) -> None:
        self.name = model_config.name
        self.datamodule = datamodule
        self.device = device
        model = build_model(
            model_config.name,
            sequence_length=datamodule.sequence_length,
            feature_count=datamodule.feature_count,
            params=model_config.params,
        )
        self.trainer = Trainer(model, training_config, device, checkpoint_path)
        fit_result = self.trainer.fit(datamodule.train_loader, datamodule.val_loader)
        self.summary = FitSummary(
            model=self.name,
            fold=fold,
            best_val_loss=fit_result.best_val_loss,
            epochs_ran=fit_result.epochs_ran,
            training_time_seconds=fit_result.training_time_seconds,
            checkpoint_path=str(fit_result.checkpoint_path),
        )

    def predict(self, raw_window: np.ndarray, previous_price: float) -> float:
        del previous_price
        scaled_window = scale_window(raw_window, self.datamodule)
        tensor = torch.as_tensor(scaled_window[None], dtype=torch.float32, device=self.device)
        self.trainer.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            predicted_scaled = float(self.trainer.model(tensor).detach().cpu().item())
        self.summary.inference_time_seconds += time.perf_counter() - start
        return predicted_scaled * self.datamodule.target_std + self.datamodule.target_mean


def fit_all_forecasters(
    datamodule: DataModule,
    training_config: TrainingConfig,
    device: torch.device,
    checkpoint_dir: Path,
    fold: int,
    seed: int,
    flat_checkpoints: bool = False,
) -> list[FittedForecaster]:
    forecasters: list[FittedForecaster] = [
        PersistenceForecaster(datamodule.target_transform, fold),
        LinearForecaster(datamodule, fold),
    ]
    for model_config in training_config.models:
        seed_everything(seed)
        reset_train_loader_seed(datamodule, seed)
        checkpoint_path = (
            checkpoint_dir / f"{model_config.name}.pt"
            if flat_checkpoints
            else checkpoint_dir / model_config.name / f"fold_{fold:04d}.pt"
        )
        forecasters.append(
            NeuralForecaster(
                model_config,
                training_config,
                datamodule,
                device,
                checkpoint_path,
                fold,
            )
        )
    return forecasters


def reset_train_loader_seed(datamodule: DataModule, seed: int) -> None:
    generator = datamodule.train_loader.generator
    if generator is None:
        msg = "The training DataLoader must have a generator for reproducible model fitting."
        raise ValueError(msg)
    generator.manual_seed(seed)


def scale_window(window: np.ndarray, datamodule: DataModule) -> np.ndarray:
    return ((window - datamodule.feature_mean) / datamodule.feature_std).astype(np.float32)


def dataset_to_numpy(dataset: object) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for feature, target in dataset:  # type: ignore[union-attr]
        features.append(feature.numpy())
        targets.append(target.numpy())
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(windows.shape[0], -1)
