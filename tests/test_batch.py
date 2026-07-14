from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import forecastle.batch as batch_module
from forecastle.batch import expand_batch_runs, run_batch


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_market_batch_expands_to_stable_single_model_runs(tmp_path) -> None:
    batch_raw = load_yaml(Path("configs/batches/markets_indicators_recursive_h20.yaml"))
    base_raw = load_yaml(Path(batch_raw["base_config"]))

    runs = expand_batch_runs(base_raw, batch_raw["batch"], tmp_path)

    assert len(runs) == 180
    assert len({run.run_id for run in runs}) == 180
    selected = next(run for run in runs if run.run_id == "wig20__cnn1d__close__seed42")
    assert [model.name for model in selected.config.training.models] == ["cnn1d"]
    assert selected.config.training.baselines == []
    assert selected.config.dataset.technical_indicators is None
    assert selected.config.dataset.horizon == 20
    assert selected.config.forecasting.strategy == "recursive"
    assert selected.config.evaluation.strategy == "walk_forward"

    baseline = next(
        run for run in runs if run.run_id == "sp500__naive_persistence__indicators__seed2026"
    )
    assert baseline.config.training.models == []
    assert baseline.config.training.baselines == ["naive_persistence"]
    assert baseline.config.dataset.technical_indicators is not None


def test_batch_resumes_and_writes_aggregate_artifacts(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "prices.csv"
    values = np.linspace(100.0, 130.0, 120)
    pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=len(values)),
            "Close": values,
        }
    ).to_csv(csv_path, index=False)

    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "output_dir": str(tmp_path / "unused")},
                "dataset": {
                    "name": "synthetic",
                    "csv_path": str(csv_path),
                    "date_column": "Date",
                    "target_column": "Close",
                    "feature_columns": ["Close"],
                    "target_transform": "log_return",
                    "sequence_length": 5,
                    "horizon": 2,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                },
                "forecasting": {"strategy": "recursive"},
                "evaluation": {
                    "strategy": "walk_forward",
                    "window": "expanding",
                    "max_folds": 1,
                },
                "training": {
                    "batch_size": 8,
                    "epochs": 1,
                    "patience": 1,
                    "models": [{"name": "mlp", "params": {"hidden_sizes": [4]}}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base_path),
                "batch": {
                    "name": "resume_test",
                    "output_dir": str(tmp_path / "batches"),
                    "datasets": [
                        {
                            "name": "synthetic",
                            "csv_path": str(csv_path),
                            "date_column": "Date",
                            "target_column": "Close",
                        }
                    ],
                    "models": ["naive_persistence"],
                    "feature_sets": {
                        "close": {
                            "feature_columns": ["Close"],
                            "technical_indicators": None,
                        },
                        "indicators": {
                            "feature_columns": ["Close"],
                            "technical_indicators": {
                                "sma_periods": [3],
                                "rsi_period": 3,
                                "macd": {
                                    "fast_period": 2,
                                    "slow_period": 4,
                                    "signal_period": 2,
                                },
                            },
                        },
                    },
                    "seeds": [7],
                    "horizon": 2,
                    "forecasting": {"strategy": "recursive"},
                    "evaluation": {
                        "strategy": "walk_forward",
                        "window": "expanding",
                        "max_folds": 1,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    calls = 0
    real_run_experiment = batch_module.run_experiment

    def counted_run(config):
        nonlocal calls
        calls += 1
        return real_run_experiment(config)

    monkeypatch.setattr(batch_module, "run_experiment", counted_run)
    batch_dir = run_batch(config_path)
    assert calls == 2

    resumed_dir = run_batch(config_path)
    assert resumed_dir == batch_dir
    assert calls == 2
    close_root = batch_dir / "runs/synthetic__naive_persistence__close__seed7"
    metadata_path = close_root / "metadata.yaml"
    metadata = load_yaml(metadata_path)
    metadata["status"] = "interrupted"
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    run_batch(config_path)
    assert calls == 3
    assert (close_root / "config.yaml").exists()
    assert load_yaml(metadata_path)["status"] == "completed"
    for filename in [
        "aggregate_metrics.csv",
        "model_rankings.csv",
        "indicator_effects.csv",
        "cross_market_comparison.csv",
        "aggregate_horizon_metrics.csv",
        "seed_stability.csv",
    ]:
        assert (batch_dir / filename).exists()
        assert (batch_dir / filename).with_suffix(".md").exists()
    for filename in [
        "model_rankings.png",
        "indicator_effects.png",
        "cross_market_comparison.png",
        "per_horizon_performance.png",
        "seed_stability.png",
    ]:
        assert (batch_dir / "plots" / filename).exists()
