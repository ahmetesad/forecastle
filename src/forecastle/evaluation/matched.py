from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from forecastle.evaluation.forecasting import format_date

if TYPE_CHECKING:
    from pathlib import Path

    from forecastle.config import DatasetConfig, ForecastStrategy
    from forecastle.data import DatasetBundle
    from forecastle.evaluation.walk_forward import WalkForwardFold


PLAN_KEY = ["fold", "forecast_origin", "target_date", "horizon_step"]
FOLD_COLUMNS = [
    "fold",
    "forecast_origin",
    "forecast_end",
    "train_samples",
    "validation_samples",
    "train_target_start",
    "train_target_end",
    "validation_target_start",
    "validation_target_end",
]


class MatchedOriginIntegrityError(ValueError):
    """Raised when a run differs from its canonical matched-origin plan."""


def load_matched_plan(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Matched-origin plan not found: {path}"
        raise FileNotFoundError(msg)
    plan = pd.read_csv(path)
    required = {*PLAN_KEY, *FOLD_COLUMNS}
    missing = sorted(required.difference(plan.columns))
    if missing:
        msg = f"Matched-origin plan {path} is missing columns: {missing}."
        raise MatchedOriginIntegrityError(msg)
    if plan.empty or plan.duplicated(PLAN_KEY).any():
        msg = f"Matched-origin plan {path} is empty or has duplicate forecast keys."
        raise MatchedOriginIntegrityError(msg)
    return _normalize_plan(plan)


def plan_origin_indices(plan: pd.DataFrame, bundle: DatasetBundle) -> list[int]:
    index_by_date = {format_date(date): index for index, date in enumerate(bundle.dates)}
    origins = plan[["fold", "forecast_origin"]].drop_duplicates().sort_values("fold")
    missing = sorted(set(origins["forecast_origin"]) - index_by_date.keys())
    if missing:
        msg = f"Matched-origin dates are absent from the usable dataset: {missing[:3]}."
        raise MatchedOriginIntegrityError(msg)
    return [index_by_date[date] for date in origins["forecast_origin"]]


def select_forecast_schedule(
    plan: pd.DataFrame,
    strategy: ForecastStrategy,
    horizon: int,
) -> pd.DataFrame:
    schedule = _normalize_plan(plan)[PLAN_KEY].drop_duplicates()
    if strategy == "direct":
        schedule = schedule[schedule["horizon_step"].eq(horizon)]
    if schedule.empty:
        msg = f"Matched-origin source contains no {strategy} horizon-{horizon} forecast rows."
        raise MatchedOriginIntegrityError(msg)
    return schedule.sort_values(PLAN_KEY).reset_index(drop=True)


def validate_forecast_schedule(
    actual: pd.DataFrame,
    source: pd.DataFrame,
    strategy: ForecastStrategy,
    horizon: int,
    context: str,
) -> None:
    actual_schedule = select_forecast_schedule(actual, strategy, horizon)
    source_schedule = select_forecast_schedule(source, strategy, horizon)
    if actual_schedule.equals(source_schedule):
        return
    msg = (
        f"Forecast schedule integrity failed for {context}: expected "
        f"{len(source_schedule)} dated forecast rows and found {len(actual_schedule)}."
    )
    raise MatchedOriginIntegrityError(msg)


def build_plan_frame(
    folds: list[WalkForwardFold],
    fold_rows: list[dict[str, Any]],
    bundle: DatasetBundle,
    dataset_config: DatasetConfig,
    strategy: ForecastStrategy = "recursive",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, fold_row in zip(folds, fold_rows, strict=True):
        available_steps = min(
            dataset_config.horizon,
            len(bundle.target_prices) - fold.origin_index - 1,
        )
        horizon_steps = (
            [dataset_config.horizon]
            if strategy == "direct" and available_steps >= dataset_config.horizon
            else range(1, available_steps + 1)
            if strategy == "recursive"
            else []
        )
        for horizon_step in horizon_steps:
            rows.append(
                {
                    **fold_row,
                    "target_date": format_date(bundle.dates[fold.origin_index + horizon_step]),
                    "horizon_step": horizon_step,
                }
            )
    return _normalize_plan(pd.DataFrame(rows))


def validate_plan_frame(actual: pd.DataFrame, expected: pd.DataFrame, context: str) -> None:
    actual_normalized = _normalize_plan(actual)
    expected_normalized = _normalize_plan(expected)
    columns = [*PLAN_KEY, *[column for column in FOLD_COLUMNS if column not in PLAN_KEY]]
    actual_view = actual_normalized[columns].sort_values(PLAN_KEY).reset_index(drop=True)
    expected_view = expected_normalized[columns].sort_values(PLAN_KEY).reset_index(drop=True)
    if actual_view.equals(expected_view):
        return
    msg = (
        f"Matched-origin integrity failed for {context}: expected {len(expected_view)} forecast "
        f"rows and found {len(actual_view)} with different dates or fold boundaries."
    )
    raise MatchedOriginIntegrityError(msg)


def _normalize_plan(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in [
        "forecast_origin",
        "target_date",
        "forecast_end",
        "train_target_start",
        "train_target_end",
        "validation_target_start",
        "validation_target_end",
    ]:
        if column in normalized:
            normalized[column] = pd.to_datetime(normalized[column]).map(format_date)
    for column in ["fold", "horizon_step", "train_samples", "validation_samples"]:
        if column in normalized:
            normalized[column] = normalized[column].astype(int)
    return normalized
