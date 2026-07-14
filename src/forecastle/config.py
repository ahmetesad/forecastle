from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
TargetTransform = Literal["price", "return", "log_return"]
TuningMetric = Literal["rmse", "price_rmse"]
ForecastStrategy = Literal["direct", "recursive"]
EvaluationStrategy = Literal["holdout", "walk_forward"]
WindowStrategy = Literal["expanding", "rolling"]
BaselineName = Literal["naive_persistence", "linear_regression"]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path = Path("outputs")
    seed: int = 42
    device: DeviceName = "auto"


@dataclass(frozen=True)
class MacdConfig:
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9


@dataclass(frozen=True)
class TechnicalIndicatorConfig:
    sma_periods: list[int] = field(default_factory=list)
    rsi_period: int | None = None
    macd: MacdConfig | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    csv_path: Path
    date_column: str
    target_column: str
    source: str = "csv"
    feature_columns: list[str] | None = None
    target_transform: TargetTransform = "log_return"
    sequence_length: int = 30
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    scale_features: bool = True
    scale_target: bool = True
    dropna: bool = True
    technical_indicators: TechnicalIndicatorConfig | None = None


@dataclass(frozen=True)
class ForecastingConfig:
    strategy: ForecastStrategy = "direct"


@dataclass(frozen=True)
class EvaluationConfig:
    strategy: EvaluationStrategy = "holdout"
    window: WindowStrategy = "expanding"
    step_size: int | None = None
    validation_size: int | None = None
    train_window_size: int | None = None
    max_folds: int | None = None


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
    baselines: list[BaselineName] = field(
        default_factory=lambda: ["naive_persistence", "linear_regression"]
    )


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
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


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
    forecasting = _parse_forecasting(raw.get("forecasting", {}))
    evaluation = _parse_evaluation(raw.get("evaluation", {}))
    if not training.models and not training.baselines:
        msg = "At least one model or baseline must be configured under training."
        raise ValueError(msg)
    _validate_ratios(dataset)
    _validate_tuning(tuning)
    validate_recursive_features(dataset, forecasting)
    return AppConfig(
        experiment=experiment,
        dataset=dataset,
        training=training,
        tuning=tuning,
        forecasting=forecasting,
        evaluation=evaluation,
    )


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
    indicators = _parse_technical_indicators(raw.get("technical_indicators"))
    return DatasetConfig(
        name=str(raw["name"]),
        source=str(raw.get("source", "csv")),
        csv_path=Path(raw["csv_path"]),
        date_column=str(raw["date_column"]),
        target_column=str(raw["target_column"]),
        feature_columns=list(feature_columns) if feature_columns is not None else None,
        target_transform=str(raw.get("target_transform", "log_return")),  # type: ignore[arg-type]
        sequence_length=int(raw.get("sequence_length", 30)),
        horizon=int(raw.get("horizon", 1)),
        train_ratio=float(raw.get("train_ratio", 0.7)),
        val_ratio=float(raw.get("val_ratio", 0.15)),
        test_ratio=float(raw.get("test_ratio", 0.15)),
        scale_features=bool(raw.get("scale_features", True)),
        scale_target=bool(raw.get("scale_target", True)),
        dropna=bool(raw.get("dropna", True)),
        technical_indicators=indicators,
    )


def _parse_technical_indicators(raw: Any) -> TechnicalIndicatorConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        msg = "dataset.technical_indicators must be a mapping."
        raise ValueError(msg)
    macd_raw = raw.get("macd")
    macd = None
    if macd_raw is not None:
        if not isinstance(macd_raw, dict):
            msg = "dataset.technical_indicators.macd must be a mapping or null."
            raise ValueError(msg)
        macd = MacdConfig(
            fast_period=int(macd_raw.get("fast_period", 12)),
            slow_period=int(macd_raw.get("slow_period", 26)),
            signal_period=int(macd_raw.get("signal_period", 9)),
        )
    rsi_period = raw.get("rsi_period")
    config = TechnicalIndicatorConfig(
        sma_periods=[int(value) for value in raw.get("sma_periods", [])],
        rsi_period=int(rsi_period) if rsi_period is not None else None,
        macd=macd,
    )
    _validate_technical_indicators(config)
    return config


def _parse_training(raw: dict[str, Any]) -> TrainingConfig:
    model_items = raw.get("models", [])
    models = [
        ModelRunConfig(name=str(item["name"]), params=dict(item.get("params", {})))
        for item in model_items
    ]
    baseline_items = raw.get("baselines", ["naive_persistence", "linear_regression"])
    baselines = [str(name) for name in baseline_items]
    unknown_baselines = sorted(set(baselines) - {"naive_persistence", "linear_regression"})
    if unknown_baselines:
        msg = f"Unknown training baselines: {', '.join(unknown_baselines)}."
        raise ValueError(msg)
    if len(baselines) != len(set(baselines)):
        msg = "training.baselines must not contain duplicates."
        raise ValueError(msg)
    return TrainingConfig(
        batch_size=int(raw.get("batch_size", 64)),
        epochs=int(raw.get("epochs", 50)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        patience=int(raw.get("patience", 8)),
        num_workers=int(raw.get("num_workers", 0)),
        models=models,
        baselines=baselines,  # type: ignore[arg-type]
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


def _parse_forecasting(raw: dict[str, Any]) -> ForecastingConfig:
    strategy = str(raw.get("strategy", "direct"))
    if strategy not in {"direct", "recursive"}:
        msg = "forecasting.strategy must be one of: direct, recursive."
        raise ValueError(msg)
    return ForecastingConfig(strategy=strategy)  # type: ignore[arg-type]


def _parse_evaluation(raw: dict[str, Any]) -> EvaluationConfig:
    strategy = str(raw.get("strategy", "holdout"))
    window = str(raw.get("window", "expanding"))
    if strategy not in {"holdout", "walk_forward"}:
        msg = "evaluation.strategy must be one of: holdout, walk_forward."
        raise ValueError(msg)
    if window not in {"expanding", "rolling"}:
        msg = "evaluation.window must be one of: expanding, rolling."
        raise ValueError(msg)
    return EvaluationConfig(
        strategy=strategy,  # type: ignore[arg-type]
        window=window,  # type: ignore[arg-type]
        step_size=_optional_positive_int(raw.get("step_size"), "evaluation.step_size"),
        validation_size=_optional_positive_int(
            raw.get("validation_size"), "evaluation.validation_size"
        ),
        train_window_size=_optional_positive_int(
            raw.get("train_window_size"), "evaluation.train_window_size"
        ),
        max_folds=_optional_positive_int(raw.get("max_folds"), "evaluation.max_folds"),
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


def _validate_technical_indicators(config: TechnicalIndicatorConfig) -> None:
    if any(period < 1 for period in config.sma_periods):
        msg = "SMA periods must be positive integers."
        raise ValueError(msg)
    if config.rsi_period is not None and config.rsi_period < 1:
        msg = "RSI period must be positive."
        raise ValueError(msg)
    if config.macd is not None:
        periods = (config.macd.fast_period, config.macd.slow_period, config.macd.signal_period)
        if any(period < 1 for period in periods):
            msg = "MACD periods must be positive."
            raise ValueError(msg)
        if config.macd.fast_period >= config.macd.slow_period:
            msg = "MACD fast_period must be smaller than slow_period."
            raise ValueError(msg)


def validate_recursive_features(
    dataset: DatasetConfig,
    forecasting: ForecastingConfig,
) -> None:
    if forecasting.strategy != "recursive":
        return
    feature_columns = dataset.feature_columns
    if feature_columns is None or feature_columns != [dataset.target_column]:
        msg = (
            "Recursive forecasting supports only the target Close as a raw feature. "
            f"Set dataset.feature_columns to [{dataset.target_column!r}]; future OHLCV and "
            "exogenous values are unavailable without leakage."
        )
        raise ValueError(msg)


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        msg = f"{name} must be positive when provided."
        raise ValueError(msg)
    return parsed
