from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from forecastle.config import TechnicalIndicatorConfig


def required_indicator_warmup(config: TechnicalIndicatorConfig | None) -> int:
    """Return the number of leading rows without a complete configured feature vector."""
    if config is None:
        return 0
    candidates = [period - 1 for period in config.sma_periods]
    if config.rsi_period is not None:
        candidates.append(config.rsi_period)
    if config.macd is not None:
        candidates.append(config.macd.slow_period + config.macd.signal_period - 2)
    return max(candidates, default=0)


def add_technical_indicators(
    frame: pd.DataFrame,
    target_column: str,
    config: TechnicalIndicatorConfig | None,
) -> tuple[pd.DataFrame, list[str]]:
    if config is None:
        return frame, []

    indicators = calculate_indicators(frame[target_column], config)
    duplicates = sorted(set(frame.columns).intersection(indicators.columns))
    if duplicates:
        msg = f"Technical indicator columns already exist in the dataset: {duplicates}"
        raise ValueError(msg)

    combined = pd.concat([frame, indicators], axis=1)
    if not indicators.empty:
        combined = combined.dropna(subset=list(indicators.columns))
    return combined, list(indicators.columns)


def calculate_indicators(
    prices: pd.Series | np.ndarray,
    config: TechnicalIndicatorConfig | None,
) -> pd.DataFrame:
    series = pd.Series(prices, dtype="float64").reset_index(drop=True)
    columns: dict[str, pd.Series] = {}
    if config is None:
        return pd.DataFrame(index=series.index)

    for period in sorted(set(config.sma_periods)):
        columns[f"SMA_{period}"] = series.rolling(window=period, min_periods=period).mean()

    if config.rsi_period is not None:
        columns[f"RSI_{config.rsi_period}"] = calculate_rsi(series, config.rsi_period)

    if config.macd is not None:
        fast = series.ewm(
            span=config.macd.fast_period,
            adjust=False,
            min_periods=config.macd.fast_period,
        ).mean()
        slow = series.ewm(
            span=config.macd.slow_period,
            adjust=False,
            min_periods=config.macd.slow_period,
        ).mean()
        macd = fast - slow
        signal = macd.ewm(
            span=config.macd.signal_period,
            adjust=False,
            min_periods=config.macd.signal_period,
        ).mean()
        columns["MACD"] = macd
        columns["MACD_signal"] = signal
        columns["MACD_histogram"] = macd - signal

    return pd.DataFrame(columns, index=series.index)


def build_close_feature_matrix(
    prices: np.ndarray,
    target_column: str,
    config: TechnicalIndicatorConfig | None,
) -> np.ndarray:
    price_series = pd.Series(prices, dtype="float64")
    frame = pd.DataFrame({target_column: price_series})
    indicators = calculate_indicators(price_series, config)
    return pd.concat([frame, indicators], axis=1).to_numpy(dtype=np.float32)


def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    changes = prices.diff()
    gains = changes.clip(lower=0.0)
    losses = -changes.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    relative_strength = average_gain / average_loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    gains_only = (average_gain > 0.0) & (average_loss == 0.0)
    losses_only = (average_gain == 0.0) & (average_loss > 0.0)
    flat = (average_gain == 0.0) & (average_loss == 0.0)
    rsi = rsi.mask(gains_only, 100.0)
    rsi = rsi.mask(losses_only, 0.0)
    return rsi.mask(flat, 50.0)
