from __future__ import annotations

from typing import TYPE_CHECKING

from forecastle.evaluation.common import (
    make_run_dir as _make_run_dir,
)
from forecastle.evaluation.common import (
    prefixed_metrics as _prefixed_metrics,
)
from forecastle.evaluation.common import (
    reconstruct_prices as _reconstruct_prices,
)
from forecastle.evaluation.common import (
    resolve_device as _resolve_device,
)
from forecastle.evaluation.common import (
    unscale as _unscale,
)
from forecastle.evaluation.registry import run_evaluation
from forecastle.evaluation.types import ExperimentResult

__all__ = [
    "ExperimentResult",
    "make_run_dir",
    "prefixed_metrics",
    "reconstruct_prices",
    "resolve_device",
    "run_experiment",
    "run_holdout_experiment",
    "unscale",
]

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import torch

    from forecastle.config import AppConfig


def run_experiment(config: AppConfig) -> ExperimentResult:
    return run_evaluation(config)


def run_holdout_experiment(config: AppConfig) -> ExperimentResult:
    """Backward-compatible entry point for the original direct holdout evaluator."""
    from forecastle.evaluation.direct_holdout import run_direct_holdout

    return run_direct_holdout(config)


def resolve_device(device_name: str) -> torch.device:
    return _resolve_device(device_name)


def make_run_dir(output_dir: Path, experiment_name: str) -> Path:
    return _make_run_dir(output_dir, experiment_name)


def unscale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return _unscale(values, mean, std)


def reconstruct_prices(
    previous_prices: np.ndarray,
    predictions: np.ndarray,
    target_transform: str,
) -> np.ndarray:
    return _reconstruct_prices(previous_prices, predictions, target_transform)


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return _prefixed_metrics(prefix, metrics)
