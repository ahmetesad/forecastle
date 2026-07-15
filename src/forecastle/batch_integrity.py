from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from forecastle.artifacts import write_dataframe
from forecastle.evaluation.matched import (
    FOLD_COLUMNS,
    MatchedOriginIntegrityError,
    load_matched_plan,
    validate_plan_frame,
)

PREDICTION_KEY = ["fold", "forecast_origin", "target_date", "horizon_step"]
PERSISTENCE_VALUE_COLUMNS = [
    "actual",
    "prediction",
    "actual_price",
    "prediction_price",
]
METRIC_COLUMNS = ["mae", "rmse", "mape", "r2", "price_mae", "price_rmse", "price_mape", "price_r2"]


def validate_run_against_plan(
    artifact_dir: Path,
    expected_model: str,
    plan_path: Path,
) -> dict[str, bool | int]:
    plan = load_matched_plan(plan_path)
    predictions_path = artifact_dir / "predictions" / f"{expected_model}_predictions.csv"
    folds_path = artifact_dir / "folds.csv"
    if not predictions_path.is_file() or not folds_path.is_file():
        msg = f"Matched run artifacts are incomplete under {artifact_dir}."
        raise FileNotFoundError(msg)

    predictions = pd.read_csv(predictions_path)
    folds = pd.read_csv(folds_path)
    if predictions.duplicated(PREDICTION_KEY).any():
        msg = f"Run {artifact_dir} contains duplicate matched-origin forecast keys."
        raise MatchedOriginIntegrityError(msg)
    fold_view = folds[FOLD_COLUMNS].drop_duplicates()
    actual = predictions[PREDICTION_KEY].merge(
        fold_view,
        on=["fold", "forecast_origin"],
        how="left",
        validate="many_to_one",
    )
    validate_plan_frame(actual, plan, str(artifact_dir))
    return {
        "origins_match": True,
        "target_dates_match": True,
        "fold_boundaries_match": True,
        "fold_counts_match": folds["fold"].nunique() == plan["fold"].nunique(),
        "prediction_counts_match": len(predictions) == len(plan),
        "horizons_match": sorted(predictions["horizon_step"].unique().tolist())
        == sorted(plan["horizon_step"].unique().tolist()),
        "fold_count": int(folds["fold"].nunique()),
        "prediction_count": len(predictions),
    }


def write_matched_origin_integrity_report(
    batch_dir: Path,
    *,
    raise_on_mismatch: bool = True,
) -> pd.DataFrame:
    planned_path = batch_dir / "planned_runs.csv"
    if not planned_path.is_file():
        return pd.DataFrame()
    planned = pd.read_csv(planned_path)
    metadata = _metadata_by_run(batch_dir)
    rows: list[dict[str, Any]] = []

    for run in planned.to_dict("records"):
        run_id = str(run["run_id"])
        item = metadata.get(run_id)
        status = str(item.get("status", "planned")) if item is not None else "planned"
        row: dict[str, Any] = {
            "scope": "run_vs_plan",
            "run_id": run_id,
            "market": run["market"],
            "model": run["model"],
            "feature_set": run["feature_set"],
            "seed": int(run["seed"]),
            "status": status,
            "checked": False,
            "integrity_pass": pd.NA,
        }
        if status == "completed" and item is not None:
            try:
                checks = validate_run_against_plan(
                    Path(str(item["artifact_dir"])),
                    str(run["model"]),
                    Path(str(run["matched_plan_path"])),
                )
            except (FileNotFoundError, MatchedOriginIntegrityError) as error:
                row.update({"checked": True, "integrity_pass": False, "failure_reason": str(error)})
            else:
                row.update(
                    {"checked": True, "integrity_pass": all(_boolean_checks(checks)), **checks}
                )
        rows.append(row)

    rows.extend(_persistence_pair_rows(planned, metadata))
    report = pd.DataFrame(rows)
    write_dataframe(batch_dir / "matched_origin_integrity.csv", report)
    mismatches = report[report["checked"].eq(True) & report["integrity_pass"].eq(False)]
    if raise_on_mismatch and not mismatches.empty:
        first = mismatches.iloc[0]
        msg = (
            "Matched-origin study integrity failed for "
            f"{first.get('run_id') or first.get('market')}: {first.get('failure_reason', '')}"
        )
        raise MatchedOriginIntegrityError(msg)
    return report


def _persistence_pair_rows(
    planned: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    persistence = planned[planned["model"] == "naive_persistence"]
    rows = []
    for (market, seed), group in persistence.groupby(["market", "seed"]):
        by_feature = {str(row["feature_set"]): row for row in group.to_dict("records")}
        close = by_feature.get("close")
        indicators = by_feature.get("indicators")
        row: dict[str, Any] = {
            "scope": "persistence_feature_pair",
            "run_id": "",
            "market": market,
            "model": "naive_persistence",
            "feature_set": "close_vs_indicators",
            "seed": int(seed),
            "checked": False,
            "integrity_pass": pd.NA,
        }
        if close is None or indicators is None:
            row.update({"status": "invalid_plan", "failure_reason": "Missing feature condition."})
            rows.append(row)
            continue
        close_meta = metadata.get(str(close["run_id"]))
        indicator_meta = metadata.get(str(indicators["run_id"]))
        statuses = {
            str(item.get("status", "planned")) if item is not None else "planned"
            for item in (close_meta, indicator_meta)
        }
        row["status"] = "completed" if statuses == {"completed"} else ",".join(sorted(statuses))
        if statuses == {"completed"} and close_meta is not None and indicator_meta is not None:
            predictions_match, metrics_match = _compare_persistence_pair(close_meta, indicator_meta)
            row.update(
                {
                    "checked": True,
                    "persistence_predictions_match": predictions_match,
                    "persistence_metrics_match": metrics_match,
                    "integrity_pass": predictions_match and metrics_match,
                }
            )
            if not predictions_match or not metrics_match:
                row["failure_reason"] = "Persistence predictions or metrics differ by feature set."
        rows.append(row)
    return rows


def _compare_persistence_pair(
    close_metadata: dict[str, Any],
    indicator_metadata: dict[str, Any],
) -> tuple[bool, bool]:
    close_dir = Path(str(close_metadata["artifact_dir"]))
    indicator_dir = Path(str(indicator_metadata["artifact_dir"]))
    prediction_name = "predictions/naive_persistence_predictions.csv"
    close_predictions = pd.read_csv(close_dir / prediction_name).sort_values(PREDICTION_KEY)
    indicator_predictions = pd.read_csv(indicator_dir / prediction_name).sort_values(PREDICTION_KEY)
    columns = [*PREDICTION_KEY, *PERSISTENCE_VALUE_COLUMNS]
    predictions_match = (
        close_predictions[columns]
        .reset_index(drop=True)
        .equals(indicator_predictions[columns].reset_index(drop=True))
    )

    close_metrics = pd.read_csv(close_dir / "comparison.csv").iloc[0]
    indicator_metrics = pd.read_csv(indicator_dir / "comparison.csv").iloc[0]
    metrics_match = bool(
        np.array_equal(
            close_metrics[METRIC_COLUMNS].to_numpy(dtype=float),
            indicator_metrics[METRIC_COLUMNS].to_numpy(dtype=float),
            equal_nan=True,
        )
    )
    return predictions_match, metrics_match


def _metadata_by_run(batch_dir: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for path in sorted((batch_dir / "runs").glob("*/metadata.yaml")):
        with path.open(encoding="utf-8") as file:
            value = yaml.safe_load(file)
        if isinstance(value, dict) and "run_id" in value:
            items[str(value["run_id"])] = value
    return items


def _boolean_checks(checks: dict[str, bool | int]) -> list[bool]:
    return [
        value for key, value in checks.items() if key.endswith("_match") and isinstance(value, bool)
    ]
