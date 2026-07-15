from __future__ import annotations

import numpy as np
import pandas as pd

from forecastle.config import (
    AppConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelRunConfig,
    TrainingConfig,
    TuningConfig,
)
from forecastle.tuning import run_tuning


def test_run_tuning_writes_artifacts(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    rows = 70
    values = np.linspace(100.0, 115.0, rows)
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows),
            "Open": values + 0.1,
            "High": values + 0.5,
            "Low": values - 0.5,
            "Close": values,
            "Volume": np.arange(rows) + 1_000,
        }
    )
    frame.to_csv(csv_path, index=False)

    config = AppConfig(
        experiment=ExperimentConfig(name="smoke", output_dir=tmp_path / "outputs", seed=11),
        dataset=DatasetConfig(
            name="synthetic",
            csv_path=csv_path,
            date_column="Date",
            target_column="Close",
            feature_columns=["Open", "High", "Low", "Close", "Volume"],
            target_transform="log_return",
            sequence_length=4,
            horizon=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        ),
        training=TrainingConfig(
            batch_size=8,
            epochs=1,
            patience=1,
            models=[ModelRunConfig(name="lstm", params={"hidden_size": 8, "num_layers": 1})],
        ),
        tuning=TuningConfig(
            enabled=True,
            model="lstm",
            trials=2,
            metric="rmse",
            seed=11,
            study_name="smoke_lstm",
            storage=f"sqlite:///{tmp_path / 'studies' / 'smoke_lstm.db'}",
            sequence_lengths=[4],
            batch_sizes=[8],
            use_pruner=False,
        ),
    )

    run_dir = run_tuning(config)

    assert (tmp_path / "studies" / "smoke_lstm.db").exists()
    assert (run_dir / "best_params.yaml").exists()
    assert (run_dir / "best_summary.yaml").exists()
    assert (run_dir / "optimization_history.csv").exists()
    assert (run_dir / "optimization_history.md").exists()
    assert (run_dir / "parameter_importance.yaml").exists()
    assert (run_dir / "tuned_config.yaml").exists()


def test_optuna_constructs_and_trains_dnfs_trial(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    rows = 75
    values = np.linspace(100.0, 115.0, rows)
    pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows),
            "Close": values,
        }
    ).to_csv(csv_path, index=False)
    config = AppConfig(
        experiment=ExperimentConfig(name="dnfs_tuning", output_dir=tmp_path / "outputs", seed=3),
        dataset=DatasetConfig(
            name="synthetic",
            csv_path=csv_path,
            date_column="Date",
            target_column="Close",
            feature_columns=["Close"],
            sequence_length=5,
            horizon=1,
        ),
        training=TrainingConfig(
            batch_size=8,
            epochs=1,
            patience=1,
            models=[
                ModelRunConfig(
                    name="dnfs",
                    params={"encoder_type": "gru", "num_rules": 4},
                )
            ],
            baselines=[],
        ),
        tuning=TuningConfig(
            enabled=True,
            model="dnfs",
            trials=1,
            seed=3,
            storage=f"sqlite:///{tmp_path / 'dnfs.db'}",
            sequence_lengths=[5],
            batch_sizes=[8],
            use_pruner=False,
        ),
    )

    run_dir = run_tuning(config)

    assert (run_dir / "best_params.yaml").exists()
    assert (run_dir / "tuned_config.yaml").exists()
