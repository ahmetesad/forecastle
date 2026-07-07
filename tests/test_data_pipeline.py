from __future__ import annotations

import numpy as np

from forecastle.config import DatasetConfig
from forecastle.data.csv_dataset import make_baseline_predictions, make_window_target


def test_price_persistence_baseline_for_log_returns_is_zero_return(tmp_path) -> None:
    config = DatasetConfig(
        name="synthetic",
        csv_path=tmp_path / "unused.csv",
        date_column="Date",
        target_column="Close",
        target_transform="log_return",
    )
    targets = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    previous_prices = np.asarray([100.0, 101.0, 99.0], dtype=np.float32)

    baseline = make_baseline_predictions(targets, previous_prices, config)

    np.testing.assert_array_equal(baseline, np.zeros_like(targets))


def test_price_persistence_baseline_for_prices_uses_previous_price(tmp_path) -> None:
    config = DatasetConfig(
        name="synthetic",
        csv_path=tmp_path / "unused.csv",
        date_column="Date",
        target_column="Close",
        target_transform="price",
    )
    targets = np.asarray([102.0, 101.0, 103.0], dtype=np.float32)
    previous_prices = np.asarray([100.0, 101.0, 99.0], dtype=np.float32)

    baseline = make_baseline_predictions(targets, previous_prices, config)

    np.testing.assert_array_equal(baseline, previous_prices)


def test_log_return_window_target_uses_full_horizon_return() -> None:
    target = make_window_target(100.0, 121.0, "log_return")

    assert target == np.log(1.21)
