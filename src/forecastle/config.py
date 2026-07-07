from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
TargetTransform = Literal["price", "return", "log_return"]
TuningMetric = Literal["rmse", "price_rmse"]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path = Path("outputs")
    seed: int = 42
    device: DeviceName = "auto"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    csv_path: Path
    date_column: str
    target_column: str
    source: str = "csv"
    feature_columns: list[str] | None = None
    target_transform: TargetTransform = "price"
    sequence_length: int = 30
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    scale_features: bool = True
    scale_target: bool = True
    dropna: bool = True


@dataclass(frozen=True)
class ModelRunConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 8
    num_workers: int = 0
    models: list[ModelRunConfig] = field(default_factory=list)


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = False
    trials: int = 50
    metric: TuningMetric = "rmse"
    seed: int | None = None
    model: str | None = None
    study_name: str | None = None
    storage: str | None = None
    n_jobs: int = 1
    use_pruner: bool = True
    sequence_lengths: list[int] = field(default_factory=lambda: [7, 14, 30, 60, 120])
    batch_sizes: list[int] = field(default_factory=lambda: [16, 32, 64, 128])


@dataclass(frozen=True)
class AppConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    training: TrainingConfig
    tuning: TuningConfig = field(default_factory=TuningConfig)


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        msg = f"Config file {path} must contain a YAML mapping."
        raise ValueError(msg)
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> AppConfig:
    experiment = _parse_experiment(raw.get("experiment", {}))
    dataset = _parse_dataset(raw.get("dataset", {}))
    training = _parse_training(raw.get("training", {}))
    tuning = _parse_tuning(raw.get("tuning", {}))
    if not training.models:
        msg = "At least one model must be configured under training.models."
        raise ValueError(msg)
    _validate_ratios(dataset)
    _validate_tuning(tuning)
    return AppConfig(experiment=experiment, dataset=dataset, training=training, tuning=tuning)


def _parse_experiment(raw: dict[str, Any]) -> ExperimentConfig:
    device = str(raw.get("device", "auto"))
    if device not in {"auto", "cpu", "cuda", "mps"}:
        msg = "experiment.device must be one of: auto, cpu, cuda, mps."
        raise ValueError(msg)
    return ExperimentConfig(
        name=str(raw["name"]),
        output_dir=Path(raw.get("output_dir", "outputs")),
        seed=int(raw.get("seed", 42)),
        device=device,  # type: ignore[arg-type]
    )


def _parse_dataset(raw: dict[str, Any]) -> DatasetConfig:
    feature_columns = raw.get("feature_columns")
    return DatasetConfig(
        name=str(raw["name"]),
        source=str(raw.get("source", "csv")),
        csv_path=Path(raw["csv_path"]),
        date_column=str(raw["date_column"]),
        target_column=str(raw["target_column"]),
        feature_columns=list(feature_columns) if feature_columns is not None else None,
        target_transform=str(raw.get("target_transform", "price")),  # type: ignore[arg-type]
        sequence_length=int(raw.get("sequence_length", 30)),
        horizon=int(raw.get("horizon", 1)),
        train_ratio=float(raw.get("train_ratio", 0.7)),
        val_ratio=float(raw.get("val_ratio", 0.15)),
        test_ratio=float(raw.get("test_ratio", 0.15)),
        scale_features=bool(raw.get("scale_features", True)),
        scale_target=bool(raw.get("scale_target", True)),
        dropna=bool(raw.get("dropna", True)),
    )


def _parse_training(raw: dict[str, Any]) -> TrainingConfig:
    model_items = raw.get("models", [])
    models = [
        ModelRunConfig(name=str(item["name"]), params=dict(item.get("params", {})))
        for item in model_items
    ]
    return TrainingConfig(
        batch_size=int(raw.get("batch_size", 64)),
        epochs=int(raw.get("epochs", 50)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        patience=int(raw.get("patience", 8)),
        num_workers=int(raw.get("num_workers", 0)),
        models=models,
    )


def _parse_tuning(raw: dict[str, Any]) -> TuningConfig:
    if not raw:
        return TuningConfig()
    metric = str(raw.get("metric", "rmse"))
    if metric not in {"rmse", "price_rmse"}:
        msg = "tuning.metric must be one of: rmse, price_rmse."
        raise ValueError(msg)

    seed = raw.get("seed")
    model = raw.get("model")
    study_name = raw.get("study_name")
    storage = raw.get("storage")
    sequence_lengths = raw.get("sequence_lengths", [7, 14, 30, 60, 120])
    batch_sizes = raw.get("batch_sizes", [16, 32, 64, 128])
    return TuningConfig(
        enabled=bool(raw.get("enabled", False)),
        trials=int(raw.get("trials", 50)),
        metric=metric,  # type: ignore[arg-type]
        seed=int(seed) if seed is not None else None,
        model=str(model) if model is not None else None,
        study_name=str(study_name) if study_name is not None else None,
        storage=str(storage) if storage is not None else None,
        n_jobs=int(raw.get("n_jobs", 1)),
        use_pruner=bool(raw.get("use_pruner", True)),
        sequence_lengths=[int(value) for value in sequence_lengths],
        batch_sizes=[int(value) for value in batch_sizes],
    )


def _validate_ratios(config: DatasetConfig) -> None:
    total = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(total - 1.0) > 1e-6:
        msg = "Dataset split ratios must sum to 1.0."
        raise ValueError(msg)
    if config.sequence_length < 1 or config.horizon < 1:
        msg = "sequence_length and horizon must be positive."
        raise ValueError(msg)
    if config.target_transform not in {"price", "return", "log_return"}:
        msg = "dataset.target_transform must be one of: price, return, log_return."
        raise ValueError(msg)


def _validate_tuning(config: TuningConfig) -> None:
    if config.trials < 1:
        msg = "tuning.trials must be positive."
        raise ValueError(msg)
    if config.n_jobs < 1:
        msg = "tuning.n_jobs must be positive."
        raise ValueError(msg)
    if not config.sequence_lengths or any(value < 1 for value in config.sequence_lengths):
        msg = "tuning.sequence_lengths must contain positive integers."
        raise ValueError(msg)
    if not config.batch_sizes or any(value < 1 for value in config.batch_sizes):
        msg = "tuning.batch_sizes must contain positive integers."
        raise ValueError(msg)
