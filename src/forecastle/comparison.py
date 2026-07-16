from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from forecastle.artifacts import write_dataframe, write_yaml
from forecastle.evaluation.matched import PLAN_KEY, select_forecast_schedule
from forecastle.plotting import plt

ComparisonKind = Literal["direct_vs_recursive", "rolling_vs_expanding"]
IDENTITY = ["market", "model", "feature_set", "seed"]
METRICS = ["price_rmse", "price_mae", "rmse", "mae"]


class BatchComparisonError(ValueError):
    """Raised when completed batches cannot be compared as configured."""


def run_comparison(config_path: Path) -> Path:
    raw = _load_yaml(config_path)
    comparison = _require_mapping(raw.get("comparison"), "comparison")
    name = str(comparison["name"])
    kind = _comparison_kind(comparison.get("kind"))
    horizon_step = int(comparison.get("horizon_step", 20))
    reference = _require_mapping(comparison.get("reference"), "comparison.reference")
    candidate = _require_mapping(comparison.get("candidate"), "comparison.candidate")
    reference_label = str(reference["label"])
    candidate_label = str(candidate["label"])
    reference_dir = _resolve_path(config_path.parent, Path(str(reference["batch_dir"])))
    candidate_dir = _resolve_path(config_path.parent, Path(str(candidate["batch_dir"])))
    output_dir = Path(comparison.get("output_dir", "outputs/comparisons")) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_manifest = _read_required(reference_dir / "manifest.csv")
    candidate_manifest = _read_required(candidate_dir / "manifest.csv")
    coverage = _coverage(reference_manifest, candidate_manifest, reference_label, candidate_label)
    write_dataframe(output_dir / "coverage.csv", coverage)

    schedule_integrity = _schedule_integrity(
        reference_dir,
        candidate_dir,
        kind,
        horizon_step,
    )
    write_dataframe(output_dir / "schedule_integrity.csv", schedule_integrity)
    if not schedule_integrity["integrity_pass"].all():
        msg = "Comparison aborted because forecast schedules do not match."
        raise BatchComparisonError(msg)

    reference_metrics = _comparison_metrics(reference_dir, kind, horizon_step, reference=True)
    candidate_metrics = _comparison_metrics(candidate_dir, kind, horizon_step, reference=False)
    paired = _pair_metrics(
        reference_metrics,
        candidate_metrics,
        reference_label,
        candidate_label,
    )
    write_dataframe(output_dir / "paired_results.csv", paired)
    summary = _summarize_pairs(paired)
    write_dataframe(output_dir / "comparison_summary.csv", summary)

    persistence = _persistence_control(paired)
    write_dataframe(output_dir / "persistence_control.csv", persistence)
    if not persistence.empty and not persistence["integrity_pass"].all():
        msg = "Comparison aborted because persistence differs across matched conditions."
        raise BatchComparisonError(msg)

    horizons = _pair_horizons(
        reference_dir,
        candidate_dir,
        kind,
        horizon_step,
        reference_label,
        candidate_label,
    )
    write_dataframe(output_dir / "horizon_comparison.csv", horizons)
    horizon_summary = _summarize_pairs(horizons, extra_groups=["horizon_step"])
    write_dataframe(output_dir / "horizon_comparison_summary.csv", horizon_summary)

    if kind == "rolling_vs_expanding":
        folds = _pair_named_frames(
            _read_required(reference_dir / "fold_results.csv"),
            _read_required(candidate_dir / "fold_results.csv"),
            [*IDENTITY, "fold"],
            reference_label,
            candidate_label,
        )
        write_dataframe(output_dir / "fold_comparison.csv", folds)
        fold_summary = _summarize_pairs(folds, extra_groups=["fold"])
        write_dataframe(output_dir / "fold_comparison_summary.csv", fold_summary)

    plots_dir = output_dir / "plots"
    _plot_model_delta(plots_dir / "model_price_rmse_delta.png", summary, candidate_label)
    _plot_horizon_delta(
        plots_dir / "price_rmse_delta_by_horizon.png",
        horizon_summary,
        candidate_label,
    )
    write_yaml(
        output_dir / "comparison_metadata.yaml",
        {
            "name": name,
            "kind": kind,
            "horizon_step": horizon_step,
            "reference_label": reference_label,
            "reference_batch_dir": str(reference_dir),
            "candidate_label": candidate_label,
            "candidate_batch_dir": str(candidate_dir),
            "paired_runs": len(paired),
            "created_at": datetime.now(tz=UTC).isoformat(),
            "ranking_metrics": ["price_rmse", "price_mae", "rmse", "mae"],
            "return_mape_used_for_ranking": False,
        },
    )
    return output_dir


def _comparison_metrics(
    batch_dir: Path,
    kind: ComparisonKind,
    horizon_step: int,
    *,
    reference: bool,
) -> pd.DataFrame:
    if kind == "direct_vs_recursive" and reference:
        frame = _read_required(batch_dir / "horizon_results.csv")
        return frame[frame["horizon_step"].eq(horizon_step)].reset_index(drop=True)
    return _read_required(batch_dir / "run_results.csv")


def _pair_horizons(
    reference_dir: Path,
    candidate_dir: Path,
    kind: ComparisonKind,
    horizon_step: int,
    reference_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    reference = _read_required(reference_dir / "horizon_results.csv")
    candidate = _read_required(candidate_dir / "horizon_results.csv")
    if kind == "direct_vs_recursive":
        reference = reference[reference["horizon_step"].eq(horizon_step)]
        candidate = candidate[candidate["horizon_step"].eq(horizon_step)]
    return _pair_named_frames(
        reference,
        candidate,
        [*IDENTITY, "horizon_step"],
        reference_label,
        candidate_label,
    )


def _pair_metrics(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    reference_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    return _pair_named_frames(
        reference,
        candidate,
        IDENTITY,
        reference_label,
        candidate_label,
    )


def _pair_named_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    keys: list[str],
    reference_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    _require_columns(reference, [*keys, *METRICS], reference_label)
    _require_columns(candidate, [*keys, *METRICS], candidate_label)
    reference_view = reference[[*keys, *METRICS]].rename(
        columns={metric: f"{reference_label}_{metric}" for metric in METRICS}
    )
    candidate_view = candidate[[*keys, *METRICS]].rename(
        columns={metric: f"{candidate_label}_{metric}" for metric in METRICS}
    )
    paired = reference_view.merge(
        candidate_view,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    for metric in METRICS:
        reference_column = f"{reference_label}_{metric}"
        candidate_column = f"{candidate_label}_{metric}"
        paired[f"{metric}_delta"] = paired[candidate_column] - paired[reference_column]
        paired[f"{metric}_delta_pct"] = 100.0 * paired[f"{metric}_delta"] / paired[reference_column]
        paired[f"{metric}_candidate_wins"] = paired[candidate_column] < paired[reference_column]
    return paired.sort_values(keys).reset_index(drop=True)


def _summarize_pairs(
    paired: pd.DataFrame,
    extra_groups: list[str] | None = None,
) -> pd.DataFrame:
    groups = ["market", "feature_set", "model", *(extra_groups or [])]
    if paired.empty:
        return pd.DataFrame(columns=groups)
    aggregations: dict[str, tuple[str, str]] = {
        "pairs": ("seed", "count"),
        "seeds": ("seed", "nunique"),
    }
    for metric in METRICS:
        aggregations[f"{metric}_delta_mean"] = (f"{metric}_delta", "mean")
        aggregations[f"{metric}_delta_std"] = (f"{metric}_delta", "std")
        aggregations[f"{metric}_delta_pct_mean"] = (f"{metric}_delta_pct", "mean")
        aggregations[f"{metric}_delta_pct_std"] = (f"{metric}_delta_pct", "std")
        aggregations[f"{metric}_candidate_wins"] = (
            f"{metric}_candidate_wins",
            "sum",
        )
    return (
        paired.groupby(groups, as_index=False)
        .agg(**aggregations)
        .sort_values([*groups[:-1], "price_rmse_delta_pct_mean", groups[-1]])
        .reset_index(drop=True)
    )


def _coverage(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    reference_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    _require_columns(reference, [*IDENTITY, "status"], reference_label)
    _require_columns(candidate, [*IDENTITY, "status"], candidate_label)
    reference_view = reference[[*IDENTITY, "status"]].rename(
        columns={"status": f"{reference_label}_status"}
    )
    candidate_view = candidate[[*IDENTITY, "status"]].rename(
        columns={"status": f"{candidate_label}_status"}
    )
    return reference_view.merge(
        candidate_view,
        on=IDENTITY,
        how="outer",
        validate="one_to_one",
    ).sort_values(IDENTITY)


def _schedule_integrity(
    reference_dir: Path,
    candidate_dir: Path,
    kind: ComparisonKind,
    horizon_step: int,
) -> pd.DataFrame:
    markets = sorted(
        path.stem.removesuffix("_plan")
        for path in (reference_dir / "matched_origins").glob("*_plan.csv")
    )
    rows = []
    for market in markets:
        reference = _read_required(reference_dir / "matched_origins" / f"{market}_plan.csv")
        candidate_path = candidate_dir / "matched_origins" / f"{market}_plan.csv"
        candidate = _read_required(candidate_path)
        reference_strategy = "direct" if kind == "direct_vs_recursive" else "recursive"
        reference_schedule = select_forecast_schedule(
            reference,
            reference_strategy,
            horizon_step,
        )
        candidate_schedule = select_forecast_schedule(
            candidate,
            "direct" if kind == "direct_vs_recursive" else "recursive",
            horizon_step,
        )
        integrity_pass = reference_schedule.equals(candidate_schedule)
        rows.append(
            {
                "market": market,
                "reference_rows": len(reference_schedule),
                "candidate_rows": len(candidate_schedule),
                "integrity_pass": integrity_pass,
                "checked_columns": ",".join(PLAN_KEY),
            }
        )
    if not rows:
        msg = f"No matched-origin plans found under {reference_dir}."
        raise BatchComparisonError(msg)
    return pd.DataFrame(rows)


def _persistence_control(paired: pd.DataFrame) -> pd.DataFrame:
    persistence = paired[paired["model"].eq("naive_persistence")].copy()
    if persistence.empty:
        return pd.DataFrame()
    delta_columns = [f"{metric}_delta" for metric in METRICS]
    persistence["max_absolute_metric_delta"] = persistence[delta_columns].abs().max(axis=1)
    persistence["integrity_pass"] = persistence["max_absolute_metric_delta"].le(1e-12)
    return persistence[[*IDENTITY, "max_absolute_metric_delta", "integrity_pass"]].reset_index(
        drop=True
    )


def _plot_model_delta(path: Path, summary: pd.DataFrame, candidate_label: str) -> None:
    if summary.empty:
        return
    grouped = summary.groupby("model")["price_rmse_delta_pct_mean"].mean().sort_values()
    _bar_plot(
        path,
        grouped,
        f"{candidate_label} price RMSE change",
        "Mean price RMSE change (%)",
    )


def _plot_horizon_delta(path: Path, summary: pd.DataFrame, candidate_label: str) -> None:
    if summary.empty or "horizon_step" not in summary:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6))
    for model, group in summary.groupby("model"):
        ordered = group.groupby("horizon_step")["price_rmse_delta_pct_mean"].mean()
        axis.plot(ordered.index, ordered.values, marker="o", label=model)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_title(f"{candidate_label} price RMSE change by horizon")
    axis.set_xlabel("Horizon step")
    axis.set_ylabel("Mean price RMSE change (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _bar_plot(path: Path, values: pd.Series, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    colors = ["#2f855a" if value < 0 else "#c53030" for value in values]
    axis.bar(values.index.astype(str), values.to_numpy(), color=colors)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        msg = f"Comparison config {path} must contain a YAML mapping."
        raise ValueError(msg)
    return raw


def _read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Required comparison artifact is missing: {path}."
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{name} must be a YAML mapping."
        raise ValueError(msg)
    return value


def _require_columns(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        msg = f"{context} is missing required columns: {missing}."
        raise BatchComparisonError(msg)


def _comparison_kind(value: Any) -> ComparisonKind:
    if value not in {"direct_vs_recursive", "rolling_vs_expanding"}:
        msg = "comparison.kind must be direct_vs_recursive or rolling_vs_expanding."
        raise ValueError(msg)
    return value


def _resolve_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path
