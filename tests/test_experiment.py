from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from forecastle.config import (
    AppConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelRunConfig,
    TrainingConfig,
)
from forecastle.experiment import run_experiment


def test_run_experiment_writes_artifacts(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    rows = 80
    values = np.linspace(100.0, 120.0, rows)
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
        experiment=ExperimentConfig(name="smoke", output_dir=tmp_path / "outputs", seed=7),
        dataset=DatasetConfig(
            name="synthetic",
            csv_path=csv_path,
            date_column="Date",
            target_column="Close",
            feature_columns=["Open", "High", "Low", "Close", "Volume"],
            sequence_length=8,
            horizon=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        ),
        training=TrainingConfig(
            batch_size=8,
            epochs=1,
            patience=1,
            models=[ModelRunConfig(name="mlp", params={"hidden_sizes": [8], "dropout": 0.0})],
        ),
    )

    result = run_experiment(config)

    assert (result.run_dir / "checkpoints" / "mlp.pt").exists()
    assert (result.run_dir / "predictions" / "mlp_predictions.csv").exists()
    assert (result.run_dir / "plots" / "mlp_predictions.png").exists()
    assert (result.run_dir / "metrics" / "mlp_metrics.yaml").exists()
    assert (result.run_dir / "metrics" / "naive_persistence_metrics.yaml").exists()
    assert (result.run_dir / "metrics" / "linear_regression_metrics.yaml").exists()
    assert (result.run_dir / "comparison.csv").exists()
    assert (result.run_dir / "comparison.md").exists()
    assert {row["model"] for row in result.comparison_rows} == {
        "linear_regression",
        "naive_persistence",
        "mlp",
    }


def test_run_experiment_trains_dnfs(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    rows = 70
    values = np.linspace(100.0, 112.0, rows)
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
        experiment=ExperimentConfig(name="dnfs_smoke", output_dir=tmp_path / "outputs", seed=9),
        dataset=DatasetConfig(
            name="synthetic",
            csv_path=csv_path,
            date_column="Date",
            target_column="Close",
            feature_columns=["Open", "High", "Low", "Close", "Volume"],
            sequence_length=6,
            horizon=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        ),
        training=TrainingConfig(
            batch_size=8,
            epochs=1,
            patience=1,
            models=[ModelRunConfig(name="dnfs", params={"num_rules": 3, "dropout": 0.0})],
        ),
    )

    result = run_experiment(config)

    assert (result.run_dir / "checkpoints" / "dnfs.pt").exists()
    assert (result.run_dir / "metrics" / "dnfs_metrics.yaml").exists()
    assert (result.run_dir / "rule_analysis" / "rule_activations.csv").exists()
    assert (result.run_dir / "plots" / "dnfs_rule_activations.png").exists()
    assert {row["model"] for row in result.comparison_rows} == {
        "linear_regression",
        "naive_persistence",
        "dnfs",
    }


def test_run_experiment_supports_one_explicit_baseline(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=70),
            "Close": np.linspace(100.0, 112.0, 70),
        }
    ).to_csv(csv_path, index=False)
    config = AppConfig(
        experiment=ExperimentConfig(name="baseline_only", output_dir=tmp_path / "outputs"),
        dataset=DatasetConfig(
            name="synthetic",
            csv_path=csv_path,
            date_column="Date",
            target_column="Close",
            feature_columns=["Close"],
            sequence_length=6,
        ),
        training=TrainingConfig(models=[], baselines=["naive_persistence"]),
    )

    result = run_experiment(config)

    assert {row["model"] for row in result.comparison_rows} == {"naive_persistence"}
    assert (result.run_dir / "metrics" / "naive_persistence_metrics.yaml").exists()


def test_run_experiment_trains_hybrid_models(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    rows = 70
    values = np.linspace(100.0, 112.0, rows)
    pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows),
            "Close": values,
        }
    ).to_csv(csv_path, index=False)
    config = AppConfig(
        experiment=ExperimentConfig(name="hybrid_smoke", output_dir=tmp_path / "outputs", seed=9),
        dataset=DatasetConfig(
            name="synthetic",
            csv_path=csv_path,
            date_column="Date",
            target_column="Close",
            feature_columns=["Close"],
            sequence_length=6,
            horizon=1,
        ),
        training=TrainingConfig(
            batch_size=8,
            epochs=1,
            patience=1,
            models=[
                ModelRunConfig(
                    name="lstm_gru",
                    params={"lstm_hidden_size": 4, "gru_hidden_size": 4},
                ),
                ModelRunConfig(
                    name="cnn_lstm",
                    params={"channels": [4], "lstm_hidden_size": 4},
                ),
            ],
        ),
    )

    result = run_experiment(config)

    assert (result.run_dir / "checkpoints" / "lstm_gru.pt").exists()
    assert (result.run_dir / "checkpoints" / "cnn_lstm.pt").exists()


def test_model_predictions_do_not_depend_on_configuration_order(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    rows = 70
    pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows),
            "Close": np.linspace(100.0, 112.0, rows),
        }
    ).to_csv(csv_path, index=False)
    models = [
        ModelRunConfig(name="mlp", params={"hidden_sizes": [4]}),
        ModelRunConfig(name="cnn1d", params={"channels": [4]}),
    ]
    config = AppConfig(
        experiment=ExperimentConfig(name="forward", output_dir=tmp_path / "outputs", seed=17),
        dataset=DatasetConfig(
            name="synthetic",
            csv_path=csv_path,
            date_column="Date",
            target_column="Close",
            feature_columns=["Close"],
            sequence_length=6,
            horizon=1,
        ),
        training=TrainingConfig(
            batch_size=8,
            epochs=2,
            patience=2,
            models=models,
        ),
    )

    forward = run_experiment(config)
    reversed_result = run_experiment(
        replace(
            config,
            experiment=replace(config.experiment, name="reversed"),
            training=replace(config.training, models=list(reversed(models))),
        )
    )

    for model in ["mlp", "cnn1d"]:
        forward_predictions = pd.read_csv(
            forward.run_dir / "predictions" / f"{model}_predictions.csv"
        )
        reversed_predictions = pd.read_csv(
            reversed_result.run_dir / "predictions" / f"{model}_predictions.csv"
        )
        np.testing.assert_allclose(
            forward_predictions["prediction"],
            reversed_predictions["prediction"],
        )
