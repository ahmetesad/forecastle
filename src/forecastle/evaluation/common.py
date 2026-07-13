from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def unscale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values.reshape(-1) * std + mean


def reconstruct_next_price(
    previous_price: float,
    prediction: float,
    target_transform: str,
) -> float:
    if target_transform == "price":
        return prediction
    if target_transform == "return":
        return previous_price * (1.0 + prediction)
    if target_transform == "log_return":
        return previous_price * float(np.exp(prediction))
    msg = f"Unknown target transform: {target_transform}"
    raise ValueError(msg)


def reconstruct_prices(
    previous_prices: np.ndarray,
    predictions: np.ndarray,
    target_transform: str,
) -> np.ndarray:
    if target_transform == "price":
        return predictions.reshape(-1)
    if target_transform == "return":
        return previous_prices.reshape(-1) * (1.0 + predictions.reshape(-1))
    if target_transform == "log_return":
        return previous_prices.reshape(-1) * np.exp(predictions.reshape(-1))
    msg = f"Unknown target transform: {target_transform}"
    raise ValueError(msg)


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def make_run_dir(output_dir: Path, experiment_name: str) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
