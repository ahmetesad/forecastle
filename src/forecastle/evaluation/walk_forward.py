from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forecastle.artifacts import (
    plot_horizon_rmse,
    plot_predictions,
    write_comparison,
    write_dataframe,
    write_yaml,
)
from forecastle.data.csv_dataset import (
    build_datamodule_from_samples,
    load_csv_dataset,
    make_windowed_samples,
    split_slices,
)
from forecastle.evaluation.common import make_run_dir, resolve_device
from forecastle.evaluation.forecasters import fit_all_forecasters
from forecastle.evaluation.forecasting import forecast_direct, forecast_recursive, format_date
from forecastle.evaluation.metrics import (
    fit_summaries_to_frame,
    records_to_frame,
    summarize_forecasts,
)
from forecastle.evaluation.types import ExperimentResult, FitSummary, ForecastRecord

if TYPE_CHECKING:
    from pathlib import Path

    from forecastle.config import AppConfig, DatasetConfig, EvaluationConfig
    from forecastle.data import DatasetBundle, WindowedSamples


@dataclass(frozen=True)
class WalkForwardFold:
    number: int
    origin_index: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    forecast_sample_index: int


def run_walk_forward(config: AppConfig) -> ExperimentResult:
    bundle = load_csv_dataset(config.dataset)
    training_horizon = 1 if config.forecasting.strategy == "recursive" else config.dataset.horizon
    fold_dataset_config = replace(config.dataset, horizon=training_horizon)
    samples = make_windowed_samples(
        bundle,
        fold_dataset_config.sequence_length,
        training_horizon,
        fold_dataset_config.target_transform,
    )
    folds = generate_walk_forward_folds(
        samples,
        len(bundle.target_prices),
        config.dataset,
        config.evaluation,
    )
    if not folds:
        msg = "Walk-forward configuration produced no evaluable folds."
        raise ValueError(msg)

    run_dir = make_run_dir(config.experiment.output_dir, config.experiment.name)
    device = resolve_device(config.experiment.device)
    records: list[ForecastRecord] = []
    summaries: list[FitSummary] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        datamodule = build_datamodule_from_samples(
            samples,
            fold_dataset_config,
            config.training,
            config.experiment.seed + fold.number * 10_000,
            fold.train_indices,
            fold.val_indices,
            np.asarray([fold.forecast_sample_index]),
            target_name=bundle.target_name,
        )
        forecasters = fit_all_forecasters(
            datamodule,
            config.training,
            device,
            run_dir / "checkpoints",
            fold.number,
            config.experiment.seed + fold.number * 10_000,
        )
        for forecaster in forecasters:
            if config.forecasting.strategy == "recursive":
                model_records = forecast_recursive(
                    forecaster,
                    bundle,
                    fold.origin_index,
                    config.dataset,
                    fold.number,
                )
            else:
                model_records = forecast_direct(
                    forecaster,
                    bundle,
                    samples,
                    fold.forecast_sample_index,
                    config.dataset,
                    fold.number,
                )
            records.extend(model_records)
            summaries.append(forecaster.summary)
        fold_rows.append(fold_to_dict(fold, samples, bundle, config.dataset))

    comparison_rows, fold_metrics, horizon_metrics = summarize_forecasts(records, summaries)
    write_walk_forward_artifacts(
        run_dir,
        config.dataset.name,
        records,
        summaries,
        comparison_rows,
        fold_metrics,
        horizon_metrics,
        fold_rows,
    )
    return ExperimentResult(run_dir=run_dir, comparison_rows=comparison_rows)


def generate_walk_forward_folds(
    samples: WindowedSamples,
    series_length: int,
    dataset_config: DatasetConfig,
    evaluation_config: EvaluationConfig,
) -> list[WalkForwardFold]:
    train_slice, val_slice, test_slice = split_slices(len(samples.features), dataset_config)
    initial_train_size = _slice_length(train_slice)
    validation_size = evaluation_config.validation_size or _slice_length(val_slice)
    rolling_train_size = evaluation_config.train_window_size or initial_train_size
    first_test_index = _slice_start(test_slice)
    origin_index = int(samples.origin_indices[first_test_index])
    step_size = evaluation_config.step_size or dataset_config.horizon
    sample_by_origin = {
        int(sample_origin): index for index, sample_origin in enumerate(samples.origin_indices)
    }

    folds = []
    fold_number = 0
    while origin_index < series_length - 1:
        if evaluation_config.max_folds is not None and fold_number >= evaluation_config.max_folds:
            break
        forecast_sample_index = sample_by_origin.get(origin_index)
        if forecast_sample_index is None:
            break
        eligible = np.flatnonzero(samples.target_indices <= origin_index)
        if len(eligible) <= validation_size:
            break
        val_indices = eligible[-validation_size:]
        train_indices = eligible[:-validation_size]
        if evaluation_config.window == "rolling":
            train_indices = train_indices[-rolling_train_size:]
        if not len(train_indices):
            break

        folds.append(
            WalkForwardFold(
                number=fold_number,
                origin_index=origin_index,
                train_indices=train_indices,
                val_indices=val_indices,
                forecast_sample_index=forecast_sample_index,
            )
        )
        fold_number += 1
        origin_index += step_size
    return folds


def fold_to_dict(
    fold: WalkForwardFold,
    samples: WindowedSamples,
    bundle: DatasetBundle,
    dataset_config: DatasetConfig,
) -> dict[str, Any]:
    final_target_index = min(fold.origin_index + dataset_config.horizon, len(bundle.dates) - 1)
    return {
        "fold": fold.number,
        "forecast_origin": format_date(bundle.dates[fold.origin_index]),
        "forecast_end": format_date(bundle.dates[final_target_index]),
        "train_samples": len(fold.train_indices),
        "validation_samples": len(fold.val_indices),
        "train_target_start": format_date(samples.target_dates[fold.train_indices[0]]),
        "train_target_end": format_date(samples.target_dates[fold.train_indices[-1]]),
        "validation_target_start": format_date(samples.target_dates[fold.val_indices[0]]),
        "validation_target_end": format_date(samples.target_dates[fold.val_indices[-1]]),
    }


def write_walk_forward_artifacts(
    run_dir: Path,
    dataset_name: str,
    records: list[ForecastRecord],
    summaries: list[FitSummary],
    comparison_rows: list[dict[str, Any]],
    fold_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    fold_rows: list[dict[str, Any]],
) -> None:
    forecast_frame = records_to_frame(records)
    for model, group in forecast_frame.groupby("model", sort=True):
        write_dataframe(run_dir / "predictions" / f"{model}_predictions.csv", group)
        metrics_payload = next(row for row in comparison_rows if row["model"] == model)
        write_yaml(run_dir / "metrics" / f"{model}_metrics.yaml", metrics_payload)
        plot_predictions(
            run_dir / "plots" / f"{model}_predictions.png",
            group["actual_price"].to_numpy(),
            group["prediction_price"].to_numpy(),
            title=f"{dataset_name} - {model} walk-forward price",
        )

    write_comparison(run_dir, comparison_rows)
    write_dataframe(run_dir / "fold_metrics.csv", fold_metrics)
    write_dataframe(run_dir / "horizon_metrics.csv", horizon_metrics)
    write_dataframe(run_dir / "folds.csv", pd.DataFrame(fold_rows))
    write_dataframe(run_dir / "fit_summaries.csv", fit_summaries_to_frame(summaries))
    plot_horizon_rmse(
        run_dir / "plots" / "horizon_rmse.png",
        horizon_metrics,
        title=f"{dataset_name} - RMSE by forecast horizon",
    )


def _slice_start(value: slice) -> int:
    return int(value.start or 0)


def _slice_length(value: slice) -> int:
    return int((value.stop or 0) - (value.start or 0))
