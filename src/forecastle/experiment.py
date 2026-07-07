from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from forecastle.artifacts import plot_predictions, write_comparison, write_predictions, write_yaml
from forecastle.data import build_datamodule
from forecastle.models import build_model
from forecastle.training import Trainer, compute_metrics
from forecastle.utils.seed import seed_everything

if TYPE_CHECKING:
    from pathlib import Path

    from forecastle.config import AppConfig


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: Path
    comparison_rows: list[dict[str, Any]]


def run_experiment(config: AppConfig) -> ExperimentResult:
    seed_everything(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    datamodule = build_datamodule(config.dataset, config.training, config.experiment.seed)
    run_dir = make_run_dir(config.experiment.output_dir, config.experiment.name)

    comparison_rows: list[dict[str, Any]] = [
        evaluate_baseline(config, datamodule, run_dir),
        evaluate_linear_regression(config, datamodule, run_dir),
    ]
    for model_config in config.training.models:
        model = build_model(
            model_config.name,
            sequence_length=datamodule.sequence_length,
            feature_count=datamodule.feature_count,
            params=model_config.params,
        )
        checkpoint_path = run_dir / "checkpoints" / f"{model_config.name}.pt"
        trainer = Trainer(model, config.training, device, checkpoint_path)
        fit_result = trainer.fit(datamodule.train_loader, datamodule.val_loader)
        actual_scaled, predicted_scaled, inference_time = trainer.predict(datamodule.test_loader)
        actual = unscale(actual_scaled.numpy(), datamodule.target_mean, datamodule.target_std)
        predicted = unscale(predicted_scaled.numpy(), datamodule.target_mean, datamodule.target_std)
        metrics = compute_metrics(actual, predicted)
        price_predicted = reconstruct_prices(
            datamodule.test_previous_prices,
            predicted,
            datamodule.target_transform,
        )
        price_metrics = compute_metrics(datamodule.test_target_prices, price_predicted)

        metrics_payload = {
            **metrics.to_dict(),
            **prefixed_metrics("price", price_metrics.to_dict()),
            "model": model_config.name,
            "best_val_loss": fit_result.best_val_loss,
            "epochs_ran": fit_result.epochs_ran,
            "training_time_seconds": fit_result.training_time_seconds,
            "inference_time_seconds": inference_time,
            "checkpoint_path": str(fit_result.checkpoint_path),
        }
        comparison_rows.append(metrics_payload)
        write_yaml(run_dir / "metrics" / f"{model_config.name}_metrics.yaml", metrics_payload)
        write_predictions(
            run_dir / "predictions" / f"{model_config.name}_predictions.csv",
            actual,
            predicted,
            datamodule.test_target_prices,
            price_predicted,
        )
        plot_predictions(
            run_dir / "plots" / f"{model_config.name}_predictions.png",
            actual,
            predicted,
            title=f"{config.dataset.name} - {model_config.name}",
        )

    write_comparison(run_dir, comparison_rows)
    return ExperimentResult(run_dir=run_dir, comparison_rows=comparison_rows)


def evaluate_baseline(config: AppConfig, datamodule: Any, run_dir: Path) -> dict[str, Any]:
    baseline_name = "naive_persistence"
    actual = datamodule.test_actuals
    predicted = datamodule.test_baseline_predictions
    metrics = compute_metrics(actual, predicted)
    price_predicted = reconstruct_prices(
        datamodule.test_previous_prices,
        predicted,
        datamodule.target_transform,
    )
    price_metrics = compute_metrics(datamodule.test_target_prices, price_predicted)
    metrics_payload = {
        **metrics.to_dict(),
        **prefixed_metrics("price", price_metrics.to_dict()),
        "model": baseline_name,
        "best_val_loss": None,
        "epochs_ran": 0,
        "training_time_seconds": 0.0,
        "inference_time_seconds": 0.0,
        "checkpoint_path": "",
    }
    write_yaml(run_dir / "metrics" / f"{baseline_name}_metrics.yaml", metrics_payload)
    write_predictions(
        run_dir / "predictions" / f"{baseline_name}_predictions.csv",
        actual,
        predicted,
        datamodule.test_target_prices,
        price_predicted,
    )
    plot_predictions(
        run_dir / "plots" / f"{baseline_name}_predictions.png",
        actual,
        predicted,
        title=f"{config.dataset.name} - {baseline_name}",
    )
    return metrics_payload


def evaluate_linear_regression(config: AppConfig, datamodule: Any, run_dir: Path) -> dict[str, Any]:
    baseline_name = "linear_regression"
    train_x, train_y = dataset_to_numpy(datamodule.train_dataset)
    test_x, _test_y = dataset_to_numpy(datamodule.test_dataset)

    import time

    start = time.perf_counter()
    regressor = LinearRegression()
    regressor.fit(flatten_windows(train_x), train_y.reshape(-1))
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    predicted_scaled = regressor.predict(flatten_windows(test_x))
    inference_time = time.perf_counter() - start

    predicted = unscale(predicted_scaled, datamodule.target_mean, datamodule.target_std)
    actual = datamodule.test_actuals
    metrics = compute_metrics(actual, predicted)
    price_predicted = reconstruct_prices(
        datamodule.test_previous_prices,
        predicted,
        datamodule.target_transform,
    )
    price_metrics = compute_metrics(datamodule.test_target_prices, price_predicted)
    metrics_payload = {
        **metrics.to_dict(),
        **prefixed_metrics("price", price_metrics.to_dict()),
        "model": baseline_name,
        "best_val_loss": None,
        "epochs_ran": 0,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "checkpoint_path": "",
    }
    write_yaml(run_dir / "metrics" / f"{baseline_name}_metrics.yaml", metrics_payload)
    write_predictions(
        run_dir / "predictions" / f"{baseline_name}_predictions.csv",
        actual,
        predicted,
        datamodule.test_target_prices,
        price_predicted,
    )
    plot_predictions(
        run_dir / "plots" / f"{baseline_name}_predictions.png",
        actual,
        predicted,
        title=f"{config.dataset.name} - {baseline_name}",
    )
    return metrics_payload


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def make_run_dir(output_dir: Path, experiment_name: str) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def unscale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values.reshape(-1) * std + mean


def dataset_to_numpy(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for feature, target in dataset:
        features.append(feature.numpy())
        targets.append(target.numpy())
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(windows.shape[0], -1)


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
