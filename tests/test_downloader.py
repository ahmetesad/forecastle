from __future__ import annotations

import sys
import types

import pandas as pd

from forecastle.data.downloader import (
    DownloadRequest,
    download_prices,
    download_prices_from_config,
    normalize_yfinance_frame,
)


def test_normalize_yfinance_frame_keeps_training_columns() -> None:
    frame = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.2, 2.2],
            "Adj Close": [1.1, 2.1],
            "Volume": [100, 200],
        },
        index=pd.date_range("2024-01-01", periods=2, name="Date"),
    )

    normalized = normalize_yfinance_frame(frame, "^GSPC")

    assert list(normalized.columns) == [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    assert len(normalized) == 2


def test_download_prices_uses_yfinance_and_writes_csv(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "sp500.csv"
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
            index=pd.date_range("2024-01-01", periods=1, name="Date"),
        )

    fake_yfinance = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    result = download_prices(
        DownloadRequest(symbol="^GSPC", output_path=output_path, start="2024-01-01")
    )

    assert result == output_path
    assert calls["tickers"] == "^GSPC"
    assert calls["progress"] is False
    assert output_path.exists()
    assert pd.read_csv(output_path)["Close"].tolist() == [1.0]


def test_download_prices_falls_back_to_ticker_history(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "wig20.csv"
    calls = {}

    def fake_download(**kwargs):
        calls["download"] = kwargs
        return pd.DataFrame()

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            calls["ticker"] = symbol

        def history(self, **kwargs):
            calls["history"] = kwargs
            return pd.DataFrame(
                {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [0]},
                index=pd.date_range("2026-07-06", periods=1, name="Date"),
            )

    fake_yfinance = types.SimpleNamespace(download=fake_download, Ticker=FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    result = download_prices(
        DownloadRequest(symbol="WIG20.WA", output_path=output_path, start="2024-01-01")
    )

    assert result == output_path
    assert calls["download"]["tickers"] == "WIG20.WA"
    assert calls["ticker"] == "WIG20.WA"
    assert calls["history"]["start"] == "2024-01-01"
    assert output_path.exists()


def test_download_prices_from_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "downloads.yaml"
    output_path = tmp_path / "wig20.csv"
    config_path.write_text(
        """
datasets:
  wig20:
    symbol: ^WIG20
    output: ignored.csv
""",
        encoding="utf-8",
    )

    def fake_download(**_kwargs):
        return pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
            index=pd.date_range("2024-01-01", periods=1, name="Date"),
        )

    fake_yfinance = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    result = download_prices_from_config(config_path, "wig20", output_path=output_path)

    assert result == output_path
    assert output_path.exists()
