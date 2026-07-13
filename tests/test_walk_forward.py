from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from forecastle.config import (
    AppConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ForecastingConfig,
    ModelRunConfig,
    TrainingConfig,
)
from forecastle.data.csv_dataset import (
    build_datamodule_from_samples,
    load_csv_dataset,
    make_windowed_samples,
)
from forecastle.evaluation.walk_forward import generate_walk_forward_folds
from forecastle.experiment import run_experiment


def write_prices(path, rows: int = 90) -> None:
    prices = np.linspace(100.0, 140.0, rows)
    pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=rows), "Close": prices}).to_csv(
        path, index=False
    )


def dataset_config(csv_path, horizon: int = 1) -> DatasetConfig:
    return DatasetConfig(
        name="synthetic",
        csv_path=csv_path,
        date_column="Date",
        target_column="Close",
        feature_columns=["Close"],
        target_transform="log_return",
        sequence_length=5,
        horizon=horizon,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )


def test_expanding_and_rolling_fold_boundaries_are_leakage_safe(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    write_prices(csv_path)
    config = dataset_config(csv_path)
    bundle = load_csv_dataset(config)
    samples = make_windowed_samples(bundle, config.sequence_length, 1, config.target_transform)

    expanding = generate_walk_forward_folds(
        samples,
        len(bundle.target_prices),
        config,
        EvaluationConfig(
            strategy="walk_forward",
            window="expanding",
            step_size=2,
            max_folds=3,
        ),
    )
    rolling = generate_walk_forward_folds(
        samples,
        len(bundle.target_prices),
        config,
        EvaluationConfig(
            strategy="walk_forward",
            window="rolling",
            step_size=2,
            train_window_size=10,
            max_folds=3,
        ),
    )

    assert len(expanding) == len(rolling) == 3
    assert [len(fold.train_indices) for fold in expanding] == [51, 53, 55]
    assert [len(fold.train_indices) for fold in rolling] == [10, 10, 10]
    for fold in [*expanding, *rolling]:
        assert samples.target_indices[fold.train_indices].max() <= fold.origin_index
        assert samples.target_indices[fold.val_indices].max() <= fold.origin_index
        assert fold.train_indices[-1] < fold.val_indices[0]


def test_recursive_training_samples_keep_configured_horizon_cadence(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    write_prices(csv_path, rows=150)
    config = dataset_config(csv_path, horizon=20)
    bundle = load_csv_dataset(config)
    one_step_samples = make_windowed_samples(
        bundle,
        config.sequence_length,
        1,
        config.target_transform,
    )

    folds = generate_walk_forward_folds(
        one_step_samples,
        len(bundle.target_prices),
        config,
        EvaluationConfig(strategy="walk_forward", max_folds=2),
    )

    assert len(folds) == 2
    assert folds[1].origin_index - folds[0].origin_index == 20


def test_fold_scalers_fit_training_samples_only(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    write_prices(csv_path)
    config = dataset_config(csv_path)
    bundle = load_csv_dataset(config)
    samples = make_windowed_samples(bundle, config.sequence_length, 1, config.target_transform)
    fold = generate_walk_forward_folds(
        samples,
        len(bundle.target_prices),
        config,
        EvaluationConfig(strategy="walk_forward", max_folds=1),
    )[0]
    datamodule = build_datamodule_from_samples(
        samples,
        config,
        TrainingConfig(batch_size=8),
        seed=7,
        train_indices=fold.train_indices,
        val_indices=fold.val_indices,
        test_indices=np.asarray([fold.forecast_sample_index]),
    )

    expected_feature_mean = samples.features[fold.train_indices].reshape(-1, 1).mean(axis=0)
    expected_target_mean = samples.targets[fold.train_indices].mean()

    np.testing.assert_allclose(datamodule.feature_mean, expected_feature_mean)
    assert datamodule.target_mean == expected_target_mean


def test_walk_forward_recursive_smoke_outputs_unique_long_form_records(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    write_prices(csv_path, rows=75)
    config = AppConfig(
        experiment=ExperimentConfig(name="walk", output_dir=tmp_path / "outputs", seed=13),
        dataset=replace(dataset_config(csv_path), horizon=3),
        training=TrainingConfig(
            batch_size=8,
            epochs=1,
            patience=1,
            models=[ModelRunConfig(name="mlp", params={"hidden_sizes": [4]})],
        ),
        forecasting=ForecastingConfig(strategy="recursive"),
        evaluation=EvaluationConfig(
            strategy="walk_forward",
            window="expanding",
            step_size=1,
            max_folds=2,
        ),
    )

    result = run_experiment(config)

    predictions = pd.read_csv(result.run_dir / "predictions" / "mlp_predictions.csv")
    key = ["model", "fold", "forecast_origin", "target_date", "horizon_step"]
    assert not predictions.duplicated(key).any()
    assert set(predictions["horizon_step"]) == {1, 2, 3}
    assert predictions["fold"].nunique() == 2
    assert (result.run_dir / "fold_metrics.csv").exists()
    assert (result.run_dir / "horizon_metrics.csv").exists()
    assert (result.run_dir / "plots" / "horizon_rmse.png").exists()


def test_walk_forward_is_reproducible_for_same_seed(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    write_prices(csv_path, rows=70)
    base = AppConfig(
        experiment=ExperimentConfig(name="run_a", output_dir=tmp_path / "outputs", seed=23),
        dataset=replace(dataset_config(csv_path), horizon=2),
        training=TrainingConfig(
            batch_size=8,
            epochs=1,
            patience=1,
            models=[ModelRunConfig(name="mlp", params={"hidden_sizes": [4]})],
        ),
        forecasting=ForecastingConfig(strategy="recursive"),
        evaluation=EvaluationConfig(strategy="walk_forward", max_folds=1),
    )

    first = run_experiment(base)
    second = run_experiment(replace(base, experiment=replace(base.experiment, name="run_b")))
    first_predictions = pd.read_csv(first.run_dir / "predictions" / "mlp_predictions.csv")
    second_predictions = pd.read_csv(second.run_dir / "predictions" / "mlp_predictions.csv")

    np.testing.assert_allclose(first_predictions["prediction"], second_predictions["prediction"])
