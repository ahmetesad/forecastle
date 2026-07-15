from __future__ import annotations

import numpy as np
import pandas as pd

from forecastle.config import DatasetConfig, MacdConfig, TechnicalIndicatorConfig
from forecastle.data.csv_dataset import load_csv_dataset
from forecastle.data.indicators import (
    calculate_indicators,
    calculate_rsi,
    required_indicator_warmup,
)


def test_rsi_edge_cases() -> None:
    increasing = pd.Series(np.arange(30, dtype=np.float64))
    decreasing = pd.Series(np.arange(30, 0, -1, dtype=np.float64))
    flat = pd.Series(np.ones(30, dtype=np.float64))

    assert calculate_rsi(increasing, 14).iloc[-1] == 100.0
    assert calculate_rsi(decreasing, 14).iloc[-1] == 0.0
    assert calculate_rsi(flat, 14).iloc[-1] == 50.0


def test_indicators_are_causal() -> None:
    config = TechnicalIndicatorConfig(
        sma_periods=[5, 10],
        rsi_period=14,
        macd=MacdConfig(),
    )
    prices = np.linspace(100.0, 150.0, 80)
    changed_future = prices.copy()
    changed_future[60:] *= 10.0

    original = calculate_indicators(prices, config)
    changed = calculate_indicators(changed_future, config)

    pd.testing.assert_frame_equal(original.iloc[:60], changed.iloc[:60])


def test_indicator_warmup_is_removed_before_dataset_use(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    rows = 80
    pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows),
            "Close": np.linspace(100.0, 120.0, rows),
        }
    ).to_csv(csv_path, index=False)
    config = DatasetConfig(
        name="synthetic",
        csv_path=csv_path,
        date_column="Date",
        target_column="Close",
        feature_columns=["Close"],
        technical_indicators=TechnicalIndicatorConfig(sma_periods=[5]),
    )

    bundle = load_csv_dataset(config)

    assert bundle.warmup_rows == 4
    assert len(bundle.features) == rows - 4
    assert bundle.feature_names == ["Close", "SMA_5"]
    assert pd.Timestamp(bundle.dates[0]) == pd.Timestamp("2024-01-05")


def test_aligned_warmup_produces_identical_feature_condition_dates(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    rows = 100
    pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows),
            "Close": np.linspace(100.0, 120.0, rows),
        }
    ).to_csv(csv_path, index=False)
    indicators = TechnicalIndicatorConfig(
        sma_periods=[5, 10, 15, 20],
        rsi_period=14,
        macd=MacdConfig(fast_period=12, slow_period=26, signal_period=9),
    )
    warmup = required_indicator_warmup(indicators)
    common = {
        "name": "synthetic",
        "csv_path": csv_path,
        "date_column": "Date",
        "target_column": "Close",
        "feature_columns": ["Close"],
        "aligned_warmup_rows": warmup,
    }

    close = load_csv_dataset(DatasetConfig(**common))
    enriched = load_csv_dataset(DatasetConfig(**common, technical_indicators=indicators))

    assert warmup == 33
    assert close.warmup_rows == enriched.warmup_rows == 33
    np.testing.assert_array_equal(close.dates, enriched.dates)
    np.testing.assert_array_equal(close.target_prices, enriched.target_prices)
