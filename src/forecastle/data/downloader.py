from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class DownloadRequest:
    symbol: str
    output_path: Path
    start: str = "2000-01-01"
    end: str | None = None
    interval: str = "1d"
    auto_adjust: bool = False


def download_prices(request: DownloadRequest) -> Path:
    import yfinance as yf

    frame = yf.download(
        tickers=request.symbol,
        start=request.start,
        end=request.end,
        interval=request.interval,
        auto_adjust=request.auto_adjust,
        progress=False,
        actions=False,
        threads=False,
    )
    if frame.empty:
        frame = yf.Ticker(request.symbol).history(
            start=request.start,
            end=request.end,
            interval=request.interval,
            auto_adjust=request.auto_adjust,
            actions=False,
        )
    normalized = normalize_yfinance_frame(frame, request.symbol)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(request.output_path, index=False)
    return request.output_path


def download_prices_from_config(
    config_path: Path,
    dataset_name: str,
    start: str = "2000-01-01",
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = False,
    output_path: Path | None = None,
) -> Path:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    dataset = _find_dataset(raw, dataset_name)
    symbol = str(dataset["symbol"])
    output = output_path or Path(dataset["output"])
    return download_prices(
        DownloadRequest(
            symbol=symbol,
            output_path=output,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )
    )


def normalize_yfinance_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        msg = (
            f"No data returned by yfinance for symbol '{symbol}'. "
            "Check the Yahoo Finance ticker and date range."
        )
        raise ValueError(msg)

    normalized = _select_symbol_columns(frame, symbol).copy()
    normalized = normalized.reset_index()
    if "Datetime" in normalized.columns and "Date" not in normalized.columns:
        normalized = normalized.rename(columns={"Datetime": "Date"})
    if "index" in normalized.columns and "Date" not in normalized.columns:
        normalized = normalized.rename(columns={"index": "Date"})

    expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    existing_columns = [column for column in expected_columns if column in normalized.columns]
    if "Date" not in existing_columns:
        msg = "Downloaded data did not contain a date-like index or column."
        raise ValueError(msg)
    if "Close" not in existing_columns:
        msg = "Downloaded data did not contain a Close column."
        raise ValueError(msg)

    return normalized[existing_columns].dropna(subset=["Date", "Close"])


def _select_symbol_columns(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    levels = range(frame.columns.nlevels)
    for level in levels:
        if symbol in frame.columns.get_level_values(level):
            selected = frame.xs(symbol, axis=1, level=level, drop_level=True)
            if isinstance(selected, pd.Series):
                return selected.to_frame()
            return selected

    if frame.columns.nlevels == 2:
        first_level_values = set(frame.columns.get_level_values(0))
        price_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if first_level_values.intersection(price_fields):
            return frame.droplevel(1, axis=1)
        return frame.droplevel(0, axis=1)

    msg = f"Could not select columns for yfinance symbol '{symbol}'."
    raise ValueError(msg)


def _find_dataset(raw: Any, dataset_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("datasets"), dict):
        msg = "Downloads config must contain a top-level 'datasets' mapping."
        raise ValueError(msg)
    dataset = raw["datasets"].get(dataset_name)
    if not isinstance(dataset, dict):
        available = ", ".join(sorted(raw["datasets"]))
        msg = f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
        raise ValueError(msg)
    if "symbol" not in dataset or "output" not in dataset:
        msg = f"Dataset '{dataset_name}' must define symbol and output."
        raise ValueError(msg)
    return dataset
