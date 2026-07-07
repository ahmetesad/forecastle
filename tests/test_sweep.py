from __future__ import annotations

from pathlib import Path

from forecastle.sweep import load_sweep_yaml


def test_wig20_sweep_config_excludes_technical_indicators() -> None:
    raw = load_sweep_yaml(Path("configs/sweeps/wig20_features_lookbacks_horizons.yaml"))
    feature_sets = raw["sweep"]["feature_sets"]

    assert feature_sets["close"] == ["Close"]
    assert feature_sets["ohlcv"] == ["Open", "High", "Low", "Close", "Volume"]
    assert "RSI" not in feature_sets["ohlcv"]


def test_market_sweep_includes_requested_markets() -> None:
    raw = load_sweep_yaml(Path("configs/sweeps/markets_lookbacks_horizons.yaml"))

    assert [dataset["name"] for dataset in raw["sweep"]["datasets"]] == [
        "wig20",
        "sp500",
        "bist100",
    ]
