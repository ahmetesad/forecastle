from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd

from forecastle.data.csv_dataset import (
    build_datamodule_from_samples,
    load_csv_dataset,
    make_windowed_samples,
    split_slices,
)
from forecastle.evaluation.common import make_run_dir, resolve_device
from forecastle.evaluation.forecasters import fit_all_forecasters
from forecastle.evaluation.forecasting import forecast_recursive, format_date
from forecastle.evaluation.metrics import summarize_forecasts
from forecastle.evaluation.rule_analysis import drain_rule_activation_rows
from forecastle.evaluation.types import ExperimentResult, FitSummary, ForecastRecord
from forecastle.evaluation.walk_forward import write_walk_forward_artifacts

if TYPE_CHECKING:
    from forecastle.config import AppConfig


def run_recursive_holdout(config: AppConfig) -> ExperimentResult:
    bundle = load_csv_dataset(config.dataset)
    one_step_config = replace(config.dataset, horizon=1)
    samples = make_windowed_samples(
        bundle,
        one_step_config.sequence_length,
        1,
        one_step_config.target_transform,
    )
    train_slice, val_slice, test_slice = split_slices(len(samples.features), one_step_config)
    datamodule = build_datamodule_from_samples(
        samples,
        one_step_config,
        config.training,
        config.experiment.seed,
        train_slice,
        val_slice,
        test_slice,
        target_name=bundle.target_name,
        feature_names=bundle.feature_names,
    )
    run_dir = make_run_dir(config.experiment.output_dir, config.experiment.name)
    forecasters = fit_all_forecasters(
        datamodule,
        config.training,
        resolve_device(config.experiment.device),
        run_dir / "checkpoints",
        0,
        config.experiment.seed,
    )

    first_test_index = int(test_slice.start or 0)
    origin_index = int(samples.origin_indices[first_test_index])
    step_size = config.evaluation.step_size or config.dataset.horizon
    records: list[ForecastRecord] = []
    fold_rows = []
    rule_activation_rows = []
    fold = 0
    while origin_index < len(bundle.target_prices) - 1:
        if config.evaluation.max_folds is not None and fold >= config.evaluation.max_folds:
            break
        for forecaster in forecasters:
            model_records = forecast_recursive(
                forecaster, bundle, origin_index, config.dataset, fold
            )
            records.extend(model_records)
            rule_activation_rows.extend(drain_rule_activation_rows(forecaster, model_records))
        final_index = min(origin_index + config.dataset.horizon, len(bundle.dates) - 1)
        fold_rows.append(
            {
                "fold": fold,
                "forecast_origin": format_date(bundle.dates[origin_index]),
                "forecast_end": format_date(bundle.dates[final_index]),
                "train_samples": len(datamodule.train_dataset),
                "validation_samples": len(datamodule.val_dataset),
            }
        )
        fold += 1
        origin_index += step_size

    summaries: list[FitSummary] = [forecaster.summary for forecaster in forecasters]
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
        rule_activation_rows,
    )
    pd.DataFrame(fold_rows).to_csv(run_dir / "holdout_origins.csv", index=False)
    return ExperimentResult(run_dir=run_dir, comparison_rows=comparison_rows)
