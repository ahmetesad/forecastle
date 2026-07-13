from __future__ import annotations

from typing import TYPE_CHECKING

from forecastle.artifacts import (
    plot_predictions,
    write_comparison,
    write_predictions,
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
from forecastle.evaluation.forecasting import forecast_direct
from forecastle.evaluation.metrics import records_to_frame, summarize_forecasts
from forecastle.evaluation.types import ExperimentResult
from forecastle.utils.seed import seed_everything

if TYPE_CHECKING:
    from forecastle.config import AppConfig


def run_direct_holdout(config: AppConfig) -> ExperimentResult:
    seed_everything(config.experiment.seed)
    bundle = load_csv_dataset(config.dataset)
    samples = make_windowed_samples(
        bundle,
        config.dataset.sequence_length,
        config.dataset.horizon,
        config.dataset.target_transform,
    )
    train_slice, val_slice, test_slice = split_slices(len(samples.features), config.dataset)
    datamodule = build_datamodule_from_samples(
        samples,
        config.dataset,
        config.training,
        config.experiment.seed,
        train_slice,
        val_slice,
        test_slice,
        target_name=bundle.target_name,
    )
    run_dir = make_run_dir(config.experiment.output_dir, config.experiment.name)
    forecasters = fit_all_forecasters(
        datamodule,
        config.training,
        resolve_device(config.experiment.device),
        run_dir / "checkpoints",
        fold=0,
        seed=config.experiment.seed,
        flat_checkpoints=True,
    )

    test_start = int(test_slice.start or 0)
    test_stop = int(test_slice.stop or len(samples.features))
    records = []
    for forecaster in forecasters:
        for sample_index in range(test_start, test_stop):
            records.extend(
                forecast_direct(
                    forecaster,
                    bundle,
                    samples,
                    sample_index,
                    config.dataset,
                    fold=0,
                )
            )

    summaries = [forecaster.summary for forecaster in forecasters]
    comparison_rows, _fold_metrics, _horizon_metrics = summarize_forecasts(records, summaries)
    forecast_frame = records_to_frame(records)
    summaries_by_model = {summary.model: summary for summary in summaries}
    for row in comparison_rows:
        model = str(row["model"])
        summary = summaries_by_model[model]
        row.pop("folds", None)
        row.pop("forecast_count", None)
        row["checkpoint_path"] = summary.checkpoint_path
        group = forecast_frame[forecast_frame["model"] == model]
        write_yaml(run_dir / "metrics" / f"{model}_metrics.yaml", row)
        write_predictions(
            run_dir / "predictions" / f"{model}_predictions.csv",
            group["actual"].to_numpy(),
            group["prediction"].to_numpy(),
            group["actual_price"].to_numpy(),
            group["prediction_price"].to_numpy(),
        )
        plot_predictions(
            run_dir / "plots" / f"{model}_predictions.png",
            group["actual"].to_numpy(),
            group["prediction"].to_numpy(),
            title=f"{config.dataset.name} - {model}",
        )
    write_comparison(run_dir, comparison_rows)
    return ExperimentResult(run_dir=run_dir, comparison_rows=comparison_rows)
