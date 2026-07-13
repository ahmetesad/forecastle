from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forecastle.data.csv_dataset import make_window_target
from forecastle.data.indicators import build_close_feature_matrix
from forecastle.evaluation.common import reconstruct_next_price
from forecastle.evaluation.types import ForecastRecord

if TYPE_CHECKING:
    from forecastle.config import DatasetConfig
    from forecastle.data import DatasetBundle, WindowedSamples
    from forecastle.evaluation.forecasters import FittedForecaster


def forecast_direct(
    forecaster: FittedForecaster,
    bundle: DatasetBundle,
    samples: WindowedSamples,
    sample_index: int,
    dataset_config: DatasetConfig,
    fold: int,
) -> list[ForecastRecord]:
    raw_window = samples.features[sample_index]
    previous_price = float(samples.previous_prices[sample_index])
    predicted_target = forecaster.predict(raw_window, previous_price)
    predicted_price = reconstruct_next_price(
        previous_price,
        predicted_target,
        dataset_config.target_transform,
    )
    return [
        ForecastRecord(
            model=forecaster.name,
            fold=fold,
            forecast_origin=format_date(samples.origin_dates[sample_index]),
            target_date=format_date(samples.target_dates[sample_index]),
            horizon_step=dataset_config.horizon,
            actual=float(samples.targets[sample_index]),
            prediction=float(predicted_target),
            actual_price=float(samples.target_prices[sample_index]),
            prediction_price=float(predicted_price),
        )
    ]


def forecast_recursive(
    forecaster: FittedForecaster,
    bundle: DatasetBundle,
    origin_index: int,
    dataset_config: DatasetConfig,
    fold: int,
) -> list[ForecastRecord]:
    available_steps = min(dataset_config.horizon, len(bundle.target_prices) - origin_index - 1)
    if available_steps <= 0:
        return []

    history_end = bundle.warmup_rows + origin_index + 1
    synthetic_prices = bundle.indicator_history_prices[:history_end].astype(np.float64).tolist()
    origin_price = float(bundle.target_prices[origin_index])
    origin_date = format_date(bundle.dates[origin_index])
    records = []

    for horizon_step in range(1, available_steps + 1):
        feature_matrix = build_close_feature_matrix(
            np.asarray(synthetic_prices, dtype=np.float64),
            dataset_config.target_column,
            dataset_config.technical_indicators,
        )
        raw_window = feature_matrix[-dataset_config.sequence_length :]
        if len(raw_window) != dataset_config.sequence_length or not np.isfinite(raw_window).all():
            msg = "Recursive feature history is too short or contains non-finite indicator values."
            raise ValueError(msg)

        previous_price = float(synthetic_prices[-1])
        predicted_one_step = forecaster.predict(raw_window, previous_price)
        predicted_price = reconstruct_next_price(
            previous_price,
            predicted_one_step,
            dataset_config.target_transform,
        )
        synthetic_prices.append(predicted_price)

        target_index = origin_index + horizon_step
        actual_price = float(bundle.target_prices[target_index])
        records.append(
            ForecastRecord(
                model=forecaster.name,
                fold=fold,
                forecast_origin=origin_date,
                target_date=format_date(bundle.dates[target_index]),
                horizon_step=horizon_step,
                actual=make_window_target(
                    origin_price,
                    actual_price,
                    dataset_config.target_transform,
                ),
                prediction=make_window_target(
                    origin_price,
                    predicted_price,
                    dataset_config.target_transform,
                ),
                actual_price=actual_price,
                prediction_price=predicted_price,
            )
        )
    return records


def format_date(value: object) -> str:
    return pd.Timestamp(value).isoformat()
