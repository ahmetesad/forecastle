from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from forecastle.data.types import DataModule, DatasetBundle
from forecastle.data.window_dataset import WindowedTimeSeriesDataset

if TYPE_CHECKING:
    from forecastle.config import DatasetConfig, TrainingConfig


def build_csv_datamodule(
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    seed: int,
) -> DataModule:
    bundle = load_csv_dataset(dataset_config)
    windows, targets, target_prices, previous_prices = make_windows(
        bundle,
        dataset_config.sequence_length,
        dataset_config.horizon,
        dataset_config.target_transform,
    )
    baseline_predictions = make_baseline_predictions(targets, previous_prices, dataset_config)
    train_slice, val_slice, test_slice = split_slices(len(windows), dataset_config)

    train_x = windows[train_slice]
    val_x = windows[val_slice]
    test_x = windows[test_slice]
    train_y = targets[train_slice]
    val_y = targets[val_slice]
    test_y = targets[test_slice]
    val_actuals = targets[val_slice].copy()
    test_actuals = targets[test_slice].copy()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    if dataset_config.scale_features:
        flat_train = train_x.reshape(-1, train_x.shape[-1])
        feature_scaler.fit(flat_train)
        train_x = _transform_windows(feature_scaler, train_x)
        val_x = _transform_windows(feature_scaler, val_x)
        test_x = _transform_windows(feature_scaler, test_x)

    target_mean = 0.0
    target_std = 1.0
    if dataset_config.scale_target:
        target_scaler.fit(train_y.reshape(-1, 1))
        target_mean = float(target_scaler.mean_[0])
        target_std = float(target_scaler.scale_[0])
        train_y = target_scaler.transform(train_y.reshape(-1, 1)).ravel()
        val_y = target_scaler.transform(val_y.reshape(-1, 1)).ravel()
        test_y = target_scaler.transform(test_y.reshape(-1, 1)).ravel()

    train_dataset = WindowedTimeSeriesDataset(train_x, train_y)
    val_dataset = WindowedTimeSeriesDataset(val_x, val_y)
    test_dataset = WindowedTimeSeriesDataset(test_x, test_y)

    import torch

    generator = torch.Generator().manual_seed(seed)

    return DataModule(
        train_loader=DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.num_workers,
            generator=generator,
        ),
        val_loader=DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
        ),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        target_mean=target_mean,
        target_std=target_std,
        val_actuals=val_actuals,
        val_target_prices=target_prices[val_slice],
        val_previous_prices=previous_prices[val_slice],
        test_actuals=test_actuals,
        test_baseline_predictions=baseline_predictions[test_slice],
        test_target_prices=target_prices[test_slice],
        test_previous_prices=previous_prices[test_slice],
        feature_count=windows.shape[-1],
        sequence_length=dataset_config.sequence_length,
        horizon=dataset_config.horizon,
        target_name=bundle.target_name,
        target_transform=dataset_config.target_transform,
    )


def load_csv_dataset(config: DatasetConfig) -> DatasetBundle:
    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        msg = f"Dataset CSV not found: {csv_path}"
        raise FileNotFoundError(msg)

    frame = pd.read_csv(csv_path)
    required_columns = {config.date_column, config.target_column}
    if config.feature_columns is not None:
        required_columns.update(config.feature_columns)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        msg = f"Missing required columns in {csv_path}: {missing}"
        raise ValueError(msg)

    frame[config.date_column] = pd.to_datetime(frame[config.date_column])
    frame = frame.sort_values(config.date_column)
    if config.dropna:
        frame = frame.dropna(subset=list(required_columns))

    feature_names = config.feature_columns or [
        column
        for column in frame.columns
        if column not in {config.date_column, config.target_column}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    prices = frame[config.target_column].to_numpy(dtype=np.float32)
    features = frame[feature_names].to_numpy(dtype=np.float32)
    targets = _make_targets(prices, config.target_transform)
    dates = frame[config.date_column].to_numpy()
    return DatasetBundle(
        features=features,
        targets=targets,
        dates=dates,
        feature_names=feature_names,
        target_name=_target_name(config.target_column, config.target_transform),
        target_prices=prices,
    )


def make_windows(
    bundle: DatasetBundle,
    sequence_length: int,
    horizon: int,
    target_transform: str = "price",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_start = len(bundle.features) - sequence_length - horizon + 1
    if max_start <= 0:
        msg = "Dataset is too short for the configured sequence_length and horizon."
        raise ValueError(msg)

    features = []
    targets = []
    target_prices = []
    previous_prices = []
    for start in range(max_start):
        end = start + sequence_length
        target_index = end + horizon - 1
        previous_index = end - 1
        previous_price = bundle.target_prices[previous_index]
        target_price = bundle.target_prices[target_index]
        features.append(bundle.features[start:end])
        targets.append(make_window_target(previous_price, target_price, target_transform))
        target_prices.append(target_price)
        previous_prices.append(previous_price)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_prices, dtype=np.float32),
        np.asarray(previous_prices, dtype=np.float32),
    )


def make_window_target(previous_price: float, target_price: float, target_transform: str) -> float:
    if target_transform == "price":
        return target_price
    if target_transform == "return":
        return (target_price - previous_price) / previous_price
    if target_transform == "log_return":
        return float(np.log(target_price / previous_price))
    msg = f"Unknown target transform: {target_transform}"
    raise ValueError(msg)


def make_baseline_predictions(
    targets: np.ndarray,
    previous_prices: np.ndarray,
    config: DatasetConfig,
) -> np.ndarray:
    if config.target_transform == "price":
        return previous_prices.astype(np.float32)
    if config.target_transform in {"return", "log_return"}:
        return np.zeros_like(targets, dtype=np.float32)
    msg = f"Unknown target transform: {config.target_transform}"
    raise ValueError(msg)


def split_slices(length: int, config: DatasetConfig) -> tuple[slice, slice, slice]:
    train_end = int(length * config.train_ratio)
    val_end = train_end + int(length * config.val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= length:
        msg = "Split ratios produce an empty train, validation, or test split."
        raise ValueError(msg)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, length)


def _transform_windows(scaler: StandardScaler, windows: np.ndarray) -> np.ndarray:
    original_shape = windows.shape
    transformed = scaler.transform(windows.reshape(-1, original_shape[-1]))
    return transformed.reshape(original_shape).astype(np.float32)


def _make_targets(prices: np.ndarray, transform: str) -> np.ndarray:
    if transform == "price":
        return prices
    previous = prices[:-1]
    current = prices[1:]
    if transform == "return":
        values = np.concatenate([[0.0], (current - previous) / previous])
    elif transform == "log_return":
        values = np.concatenate([[0.0], np.log(current / previous)])
    else:
        msg = f"Unknown target transform: {transform}"
        raise ValueError(msg)
    return values.astype(np.float32)


def _target_name(target_column: str, transform: str) -> str:
    if transform == "price":
        return target_column
    return f"{transform}_{target_column}"
