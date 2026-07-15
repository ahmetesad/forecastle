from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from forecastle.config import DatasetConfig, parse_config
from forecastle.data.csv_dataset import load_csv_dataset, make_windowed_samples
from forecastle.evaluation.forecasters import PersistenceForecaster
from forecastle.evaluation.forecasting import forecast_direct, forecast_recursive
from forecastle.evaluation.types import FitSummary


@dataclass
class ConstantForecaster:
    name: str
    value: float
    price_increment: float | None = None

    def __post_init__(self) -> None:
        self.summary = FitSummary(model=self.name, fold=0)

    def predict(self, raw_window: np.ndarray, previous_price: float) -> float:
        del raw_window
        if self.price_increment is not None:
            return previous_price + self.price_increment
        return self.value


def write_close_csv(path, future_multiplier: float = 1.0) -> None:
    prices = np.arange(100.0, 130.0)
    prices[16:] *= future_multiplier
    pd.DataFrame(
        {"Date": pd.date_range("2024-01-01", periods=len(prices)), "Close": prices}
    ).to_csv(path, index=False)


@pytest.mark.parametrize(
    ("target_transform", "prediction", "expected"),
    [
        ("price", 0.0, [116.0, 117.0, 118.0]),
        ("return", 0.1, [126.5, 139.15, 153.065]),
        ("log_return", float(np.log(1.1)), [126.5, 139.15, 153.065]),
    ],
)
def test_recursive_forecast_feeds_predictions_back(
    tmp_path,
    target_transform: str,
    prediction: float,
    expected: list[float],
) -> None:
    csv_path = tmp_path / "prices.csv"
    write_close_csv(csv_path)
    config = DatasetConfig(
        name="synthetic",
        csv_path=csv_path,
        date_column="Date",
        target_column="Close",
        feature_columns=["Close"],
        target_transform=target_transform,  # type: ignore[arg-type]
        sequence_length=4,
        horizon=3,
    )
    bundle = load_csv_dataset(config)
    forecaster = ConstantForecaster(
        "constant",
        prediction,
        price_increment=1.0 if target_transform == "price" else None,
    )

    records = forecast_recursive(forecaster, bundle, 15, config, fold=0)

    np.testing.assert_allclose([record.prediction_price for record in records], expected)


def test_recursive_persistence_uses_same_reconstruction_engine(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    write_close_csv(csv_path)
    config = DatasetConfig(
        name="synthetic",
        csv_path=csv_path,
        date_column="Date",
        target_column="Close",
        feature_columns=["Close"],
        target_transform="log_return",
        sequence_length=4,
        horizon=3,
    )
    bundle = load_csv_dataset(config)

    records = forecast_recursive(
        PersistenceForecaster("log_return", fold=0),
        bundle,
        15,
        config,
        fold=0,
    )

    assert [record.prediction for record in records] == [0.0, 0.0, 0.0]
    assert [record.prediction_price for record in records] == [115.0, 115.0, 115.0]


def test_recursive_predictions_do_not_use_future_ground_truth(tmp_path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    write_close_csv(first_path)
    write_close_csv(second_path, future_multiplier=20.0)
    base = {
        "name": "synthetic",
        "date_column": "Date",
        "target_column": "Close",
        "feature_columns": ["Close"],
        "target_transform": "price",
        "sequence_length": 4,
        "horizon": 3,
    }
    first_config = DatasetConfig(csv_path=first_path, **base)
    second_config = DatasetConfig(csv_path=second_path, **base)
    forecaster = ConstantForecaster("constant", 0.0, price_increment=1.0)

    first = forecast_recursive(
        forecaster,
        load_csv_dataset(first_config),
        15,
        first_config,
        fold=0,
    )
    second = forecast_recursive(
        forecaster,
        load_csv_dataset(second_config),
        15,
        second_config,
        fold=0,
    )

    assert [record.prediction_price for record in first] == [
        record.prediction_price for record in second
    ]


@pytest.mark.parametrize("prediction", [-1_000.0, 1_000.0])
def test_recursive_forecast_reports_numerical_divergence(tmp_path, prediction: float) -> None:
    csv_path = tmp_path / "prices.csv"
    write_close_csv(csv_path)
    config = DatasetConfig(
        name="synthetic",
        csv_path=csv_path,
        date_column="Date",
        target_column="Close",
        feature_columns=["Close"],
        target_transform="log_return",
        sequence_length=4,
        horizon=3,
    )
    bundle = load_csv_dataset(config)

    with pytest.raises(
        ValueError,
        match=r"model=constant, fold=9, .*horizon_step=1, .*predicted_target=",
    ):
        forecast_recursive(
            ConstantForecaster("constant", prediction),
            bundle,
            15,
            config,
            fold=9,
        )


def test_direct_forecast_emits_only_endpoint(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    write_close_csv(csv_path)
    config = DatasetConfig(
        name="synthetic",
        csv_path=csv_path,
        date_column="Date",
        target_column="Close",
        feature_columns=["Close"],
        target_transform="log_return",
        sequence_length=4,
        horizon=3,
    )
    bundle = load_csv_dataset(config)
    samples = make_windowed_samples(bundle, 4, 3, "log_return")

    records = forecast_direct(
        ConstantForecaster("constant", 0.0),
        bundle,
        samples,
        sample_index=0,
        dataset_config=config,
        fold=0,
    )

    assert len(records) == 1
    assert records[0].horizon_step == 3


def test_recursive_config_rejects_ohlcv_features() -> None:
    raw = {
        "experiment": {"name": "invalid"},
        "dataset": {
            "name": "synthetic",
            "csv_path": "unused.csv",
            "date_column": "Date",
            "target_column": "Close",
            "feature_columns": ["Open", "Close"],
        },
        "training": {"models": [{"name": "mlp"}]},
        "forecasting": {"strategy": "recursive"},
    }

    with pytest.raises(ValueError, match="Recursive forecasting supports only"):
        parse_config(raw)
