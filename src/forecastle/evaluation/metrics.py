from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forecastle.evaluation.common import prefixed_metrics
from forecastle.training import compute_metrics

if TYPE_CHECKING:
    from forecastle.evaluation.types import FitSummary, ForecastRecord


FORECAST_KEY = ["model", "fold", "forecast_origin", "target_date", "horizon_step"]


def records_to_frame(records: list[ForecastRecord]) -> pd.DataFrame:
    frame = pd.DataFrame([record.to_dict() for record in records])
    if frame.empty:
        return frame
    if frame.duplicated(FORECAST_KEY).any():
        duplicates = frame.loc[frame.duplicated(FORECAST_KEY, keep=False), FORECAST_KEY]
        msg = f"Forecast records contain duplicate keys: {duplicates.to_dict(orient='records')}"
        raise ValueError(msg)
    return frame.sort_values(FORECAST_KEY).reset_index(drop=True)


def summarize_forecasts(
    records: list[ForecastRecord],
    summaries: list[FitSummary],
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    frame = records_to_frame(records)
    if frame.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    summaries_by_model: dict[str, list[FitSummary]] = defaultdict(list)
    for summary in summaries:
        summaries_by_model[summary.model].append(summary)

    comparison = []
    for model, group in frame.groupby("model", sort=True):
        row = metric_row(group)
        fit_rows = summaries_by_model[model]
        val_losses = [item.best_val_loss for item in fit_rows if item.best_val_loss is not None]
        row.update(
            {
                "model": model,
                "best_val_loss": float(np.mean(val_losses)) if val_losses else None,
                "epochs_ran": sum(item.epochs_ran for item in fit_rows),
                "training_time_seconds": sum(item.training_time_seconds for item in fit_rows),
                "inference_time_seconds": sum(item.inference_time_seconds for item in fit_rows),
                "checkpoint_path": _checkpoint_summary(fit_rows),
                "folds": int(group["fold"].nunique()),
                "forecast_count": len(group),
            }
        )
        comparison.append(row)

    fold_rows = []
    for (model, fold), group in frame.groupby(["model", "fold"], sort=True):
        fold_rows.append({"model": model, "fold": int(fold), **metric_row(group)})

    horizon_rows = []
    for (model, horizon_step), group in frame.groupby(["model", "horizon_step"], sort=True):
        horizon_rows.append(
            {
                "model": model,
                "horizon_step": int(horizon_step),
                "forecast_count": len(group),
                **metric_row(group),
            }
        )
    return comparison, pd.DataFrame(fold_rows), pd.DataFrame(horizon_rows)


def metric_row(frame: pd.DataFrame) -> dict[str, float]:
    target_metrics = compute_metrics(frame["actual"].to_numpy(), frame["prediction"].to_numpy())
    price_metrics = compute_metrics(
        frame["actual_price"].to_numpy(),
        frame["prediction_price"].to_numpy(),
    )
    return {
        **target_metrics.to_dict(),
        **prefixed_metrics("price", price_metrics.to_dict()),
    }


def fit_summaries_to_frame(summaries: list[FitSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": summary.model,
                "fold": summary.fold,
                "best_val_loss": summary.best_val_loss,
                "epochs_ran": summary.epochs_ran,
                "training_time_seconds": summary.training_time_seconds,
                "inference_time_seconds": summary.inference_time_seconds,
                "checkpoint_path": summary.checkpoint_path,
            }
            for summary in summaries
        ]
    )


def _checkpoint_summary(summaries: list[FitSummary]) -> str:
    paths = [summary.checkpoint_path for summary in summaries if summary.checkpoint_path]
    if not paths:
        return ""
    from pathlib import Path

    return str(Path(paths[0]).parent)
