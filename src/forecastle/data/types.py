from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    from torch.utils.data import DataLoader, Dataset

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetBundle:
    features: np.ndarray
    targets: np.ndarray
    dates: np.ndarray
    feature_names: list[str]
    target_name: str
    target_prices: np.ndarray
    indicator_history_prices: np.ndarray
    warmup_rows: int


@dataclass(frozen=True)
class WindowedSamples:
    features: np.ndarray
    targets: np.ndarray
    target_prices: np.ndarray
    previous_prices: np.ndarray
    origin_dates: np.ndarray
    target_dates: np.ndarray
    origin_indices: np.ndarray
    target_indices: np.ndarray


@dataclass(frozen=True)
class DataModule:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    target_mean: float
    target_std: float
    val_actuals: np.ndarray
    val_target_prices: np.ndarray
    val_previous_prices: np.ndarray
    test_actuals: np.ndarray
    test_baseline_predictions: np.ndarray
    test_target_prices: np.ndarray
    test_previous_prices: np.ndarray
    feature_count: int
    sequence_length: int
    horizon: int
    target_name: str
    target_transform: str
    feature_mean: np.ndarray
    feature_std: np.ndarray
    val_target_dates: np.ndarray
    test_target_dates: np.ndarray
    val_origin_dates: np.ndarray
    test_origin_dates: np.ndarray
