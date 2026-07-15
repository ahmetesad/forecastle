from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forecastle.artifacts import write_dataframe
from forecastle.batch_integrity import (
    validate_run_against_plan,
    write_matched_origin_integrity_report,
)
from forecastle.plotting import plt

SUMMARY_METRICS = [
    "price_rmse",
    "price_mae",
    "rmse",
    "mae",
    "training_time_seconds",
    "inference_time_seconds",
]


def write_batch_summaries(batch_dir: Path) -> None:
    manifest, results, horizons, folds = _collect_completed_results(batch_dir)
    write_dataframe(batch_dir / "manifest.csv", manifest)
    write_dataframe(batch_dir / "run_results.csv", results)
    write_dataframe(batch_dir / "horizon_results.csv", horizons)
    write_dataframe(batch_dir / "fold_results.csv", folds)

    completion = _completion_summary(manifest)
    divergence = _divergence_frequency(manifest)
    write_dataframe(batch_dir / "completion_summary.csv", completion)
    write_dataframe(batch_dir / "divergence_frequency.csv", divergence)
    if "matched_plan_path" in manifest and manifest["matched_plan_path"].fillna("").ne("").any():
        write_matched_origin_integrity_report(batch_dir)

    if not results.empty:
        results = _add_persistence_comparisons(results)
    if not horizons.empty:
        horizons = _add_horizon_persistence_comparisons(horizons)
    write_dataframe(batch_dir / "run_results.csv", results)
    write_dataframe(batch_dir / "horizon_results.csv", horizons)

    aggregate = _aggregate_metrics(results, completion)
    rankings = _aggregate_rankings(results, completion)
    indicator_pairs, indicator_summary = _indicator_effects(results)
    horizon_summary = _aggregate_horizons(horizons)
    fold_summary = _aggregate_folds(folds)
    seed_stability = _seed_stability(results)

    write_dataframe(batch_dir / "aggregate_metrics.csv", aggregate)
    write_dataframe(batch_dir / "model_rankings.csv", rankings)
    write_dataframe(batch_dir / "indicator_effects.csv", indicator_pairs)
    write_dataframe(batch_dir / "indicator_effect_summary.csv", indicator_summary)
    write_dataframe(batch_dir / "cross_market_comparison.csv", aggregate)
    write_dataframe(batch_dir / "aggregate_horizon_metrics.csv", horizon_summary)
    write_dataframe(batch_dir / "aggregate_fold_metrics.csv", fold_summary)
    write_dataframe(batch_dir / "seed_stability.csv", seed_stability)

    if results.empty:
        return
    plots_dir = batch_dir / "plots"
    _plot_model_rankings(plots_dir / "model_rankings.png", results)
    _plot_indicator_effects(plots_dir / "indicator_effects.png", indicator_pairs)
    _plot_cross_market(plots_dir / "cross_market_comparison.png", results)
    _plot_horizons(plots_dir / "per_horizon_performance.png", horizons)
    _plot_seed_stability(plots_dir / "seed_stability.png", results)


def validate_run_artifacts(
    artifact_dir: Path,
    expected_model: str,
    *,
    matched_plan_path: Path | None = None,
) -> None:
    comparison_path = artifact_dir / "comparison.csv"
    horizon_path = artifact_dir / "horizon_metrics.csv"
    if not comparison_path.is_file() or not horizon_path.is_file():
        msg = f"Run artifacts are incomplete under {artifact_dir}."
        raise FileNotFoundError(msg)
    comparison = pd.read_csv(comparison_path)
    models = comparison["model"].astype(str).tolist()
    if models != [expected_model]:
        msg = f"Expected only model {expected_model!r}, found {models}."
        raise ValueError(msg)
    if matched_plan_path is not None:
        validate_run_against_plan(artifact_dir, expected_model, matched_plan_path)


def _collect_completed_results(
    batch_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_rows = []
    result_frames = []
    horizon_frames = []
    fold_frames = []
    for metadata_path in sorted((batch_dir / "runs").glob("*/metadata.yaml")):
        with metadata_path.open(encoding="utf-8") as file:
            metadata = yaml.safe_load(file)
        if not isinstance(metadata, dict):
            continue
        manifest_rows.append(metadata)
        if metadata.get("status") != "completed":
            continue
        artifact_dir = Path(str(metadata["artifact_dir"]))
        model = str(metadata["model"])
        try:
            validate_run_artifacts(artifact_dir, model)
        except (FileNotFoundError, ValueError):
            continue
        identity = {
            "run_id": metadata["run_id"],
            "market": metadata["market"],
            "model": model,
            "feature_set": metadata["feature_set"],
            "seed": int(metadata["seed"]),
            "artifact_dir": str(artifact_dir),
        }
        comparison = pd.read_csv(artifact_dir / "comparison.csv")
        for name, value in identity.items():
            comparison[name] = value
        result_frames.append(comparison)
        horizons = pd.read_csv(artifact_dir / "horizon_metrics.csv")
        for name, value in identity.items():
            horizons[name] = value
        horizon_frames.append(horizons)
        folds_path = artifact_dir / "fold_metrics.csv"
        if folds_path.is_file():
            folds = pd.read_csv(folds_path)
            for name, value in identity.items():
                folds[name] = value
            fold_frames.append(folds)
    manifest = _durable_manifest(batch_dir, manifest_rows)
    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    horizons = pd.concat(horizon_frames, ignore_index=True) if horizon_frames else pd.DataFrame()
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    return manifest, results, horizons, folds


def _durable_manifest(batch_dir: Path, metadata_rows: list[dict[str, object]]) -> pd.DataFrame:
    planned_path = batch_dir / "planned_runs.csv"
    planned = pd.read_csv(planned_path) if planned_path.is_file() else pd.DataFrame()
    metadata = pd.DataFrame(metadata_rows)
    if planned.empty:
        return metadata
    if metadata.empty:
        manifest = planned.copy()
        manifest["status"] = "planned"
        manifest["last_action"] = "planned"
        manifest["divergence"] = False
        return manifest
    duplicate_columns = [
        column
        for column in [
            "market",
            "model",
            "feature_set",
            "seed",
            "config_sha256",
            "matched_plan_path",
        ]
        if column in metadata
    ]
    metadata = metadata.drop(columns=duplicate_columns)
    manifest = planned.merge(metadata, on="run_id", how="left", validate="one_to_one")
    manifest["status"] = manifest["status"].fillna("planned")
    manifest["last_action"] = manifest["last_action"].fillna("planned")
    if "divergence" not in manifest:
        manifest["divergence"] = False
    manifest["divergence"] = manifest["divergence"].fillna(False).astype(bool)
    return manifest


def _add_persistence_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["market", "feature_set", "seed"]
    reference = frame[frame["model"] == "naive_persistence"][[*keys, "price_rmse", "rmse"]]
    reference = reference.rename(
        columns={"price_rmse": "persistence_price_rmse", "rmse": "persistence_rmse"}
    )
    enriched = frame.drop(
        columns=[
            "persistence_price_rmse",
            "persistence_rmse",
            "price_rmse_ratio_to_persistence",
            "rmse_ratio_to_persistence",
            "price_rmse_rank",
        ],
        errors="ignore",
    ).merge(reference, on=keys, how="left", validate="many_to_one")
    enriched["price_rmse_ratio_to_persistence"] = (
        enriched["price_rmse"] / enriched["persistence_price_rmse"]
    )
    enriched["rmse_ratio_to_persistence"] = enriched["rmse"] / enriched["persistence_rmse"]
    enriched["price_rmse_rank"] = enriched.groupby(keys)["price_rmse"].rank(method="min")
    return enriched.sort_values([*keys, "price_rmse_rank", "model"]).reset_index(drop=True)


def _add_horizon_persistence_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = ["market", "feature_set", "seed", "horizon_step"]
    reference = frame[frame["model"] == "naive_persistence"][[*keys, "price_rmse"]].rename(
        columns={"price_rmse": "persistence_price_rmse"}
    )
    enriched = frame.drop(
        columns=["persistence_price_rmse", "price_rmse_ratio_to_persistence"], errors="ignore"
    ).merge(reference, on=keys, how="left", validate="many_to_one")
    enriched["price_rmse_ratio_to_persistence"] = (
        enriched["price_rmse"] / enriched["persistence_price_rmse"]
    )
    return enriched.sort_values([*keys, "model"]).reset_index(drop=True)


def _aggregate_metrics(frame: pd.DataFrame, completion: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "runs": ("run_id", "count"),
        "seeds_completed": ("seed", "nunique"),
    }
    for metric in SUMMARY_METRICS:
        if metric in frame:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_std"] = (metric, "std")
    if frame.empty:
        aggregate = pd.DataFrame(columns=["market", "feature_set", "model"])
    else:
        aggregations["seeds_beating_persistence"] = (
            "price_rmse_ratio_to_persistence",
            lambda values: int((values < 1.0).sum()),
        )
        aggregate = frame.groupby(["market", "feature_set", "model"], as_index=False).agg(
            **aggregations
        )
    aggregate = completion.merge(
        aggregate,
        on=["market", "feature_set", "model"],
        how="left",
        validate="one_to_one",
    )
    for column in ["price_rmse_mean", "price_rmse_std", "seeds_beating_persistence"]:
        if column not in aggregate:
            aggregate[column] = np.nan
    aggregate["eligible_for_ranking"] = aggregate["completion_state"].eq("complete")
    aggregate["price_rmse_rank"] = pd.NA
    eligible = aggregate["eligible_for_ranking"] & aggregate["price_rmse_mean"].notna()
    aggregate.loc[eligible, "price_rmse_rank"] = (
        aggregate[eligible].groupby(["market", "feature_set"])["price_rmse_mean"].rank(method="min")
    )
    return aggregate.sort_values(
        ["market", "feature_set", "eligible_for_ranking", "price_rmse_mean", "model"],
        ascending=[True, True, False, True, True],
        na_position="last",
    )


def _aggregate_rankings(frame: pd.DataFrame, completion: pd.DataFrame) -> pd.DataFrame:
    status = completion.groupby(["feature_set", "model"], as_index=False).agg(
        combinations=("market", "count"),
        complete_combinations=(
            "completion_state",
            lambda values: int((values == "complete").sum()),
        ),
        incomplete_combinations=(
            "completion_state",
            lambda values: int((values == "incomplete").sum()),
        ),
        failed_combinations=("completion_state", lambda values: int((values == "failed").sum())),
        planned_runs=("planned_runs", "sum"),
        completed_runs=("completed_runs", "sum"),
        failed_runs=("failed_runs", "sum"),
        divergence_count=("divergence_count", "sum"),
    )
    if frame.empty:
        return status
    ranking = frame.groupby(["feature_set", "model"], as_index=False).agg(
        mean_price_rmse_rank=("price_rmse_rank", "mean"),
        std_price_rmse_rank=("price_rmse_rank", "std"),
        mean_ratio_to_persistence=("price_rmse_ratio_to_persistence", "mean"),
        runs=("run_id", "count"),
    )
    return status.merge(ranking, on=["feature_set", "model"], how="left").sort_values(
        ["feature_set", "failed_combinations", "mean_price_rmse_rank", "model"]
    )


def _completion_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    summary = manifest.groupby(["market", "feature_set", "model"], as_index=False).agg(
        planned_runs=("run_id", "count"),
        completed_runs=("status", lambda values: int((values == "completed").sum())),
        failed_runs=("status", lambda values: int((values == "failed").sum())),
        running_runs=("status", lambda values: int((values == "running").sum())),
        planned_pending_runs=("status", lambda values: int((values == "planned").sum())),
        skipped_runs=("last_action", lambda values: int((values == "skipped").sum())),
        divergence_count=("divergence", lambda values: int(pd.Series(values).fillna(False).sum())),
    )
    summary["divergence_rate"] = summary["divergence_count"] / summary["planned_runs"]
    summary["completion_state"] = np.select(
        [
            summary["completed_runs"].eq(summary["planned_runs"]),
            summary["completed_runs"].eq(0) & summary["failed_runs"].gt(0),
        ],
        ["complete", "failed"],
        default="incomplete",
    )
    return summary.sort_values(["market", "feature_set", "model"])


def _divergence_frequency(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    return (
        manifest.groupby(["market", "feature_set", "model"], as_index=False)
        .agg(
            planned_runs=("run_id", "count"),
            failed_runs=("status", lambda values: int((values == "failed").sum())),
            divergence_count=(
                "divergence",
                lambda values: int(pd.Series(values).fillna(False).sum()),
            ),
        )
        .assign(divergence_rate=lambda value: value["divergence_count"] / value["planned_runs"])
        .sort_values(
            ["market", "feature_set", "divergence_count", "model"],
            ascending=[True, True, False, True],
        )
    )


def _indicator_effects(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["market", "model", "seed"]
    close = frame[frame["feature_set"] == "close"][[*keys, "price_rmse", "rmse"]].rename(
        columns={"price_rmse": "close_price_rmse", "rmse": "close_rmse"}
    )
    indicators = frame[frame["feature_set"] == "indicators"][[*keys, "price_rmse", "rmse"]].rename(
        columns={"price_rmse": "indicator_price_rmse", "rmse": "indicator_rmse"}
    )
    paired = close.merge(indicators, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        return paired, pd.DataFrame()
    paired["price_rmse_delta"] = paired["indicator_price_rmse"] - paired["close_price_rmse"]
    paired["price_rmse_delta_pct"] = 100.0 * paired["price_rmse_delta"] / paired["close_price_rmse"]
    paired["rmse_delta"] = paired["indicator_rmse"] - paired["close_rmse"]
    paired["rmse_delta_pct"] = 100.0 * paired["rmse_delta"] / paired["close_rmse"]
    summary = (
        paired.groupby(["market", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            price_rmse_delta_pct_mean=("price_rmse_delta_pct", "mean"),
            price_rmse_delta_pct_std=("price_rmse_delta_pct", "std"),
            rmse_delta_pct_mean=("rmse_delta_pct", "mean"),
            rmse_delta_pct_std=("rmse_delta_pct", "std"),
        )
        .sort_values(["market", "price_rmse_delta_pct_mean", "model"])
    )
    return paired, summary


def _aggregate_horizons(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby(["market", "feature_set", "model", "horizon_step"], as_index=False)
        .agg(
            seeds_completed=("seed", "nunique"),
            price_rmse_mean=("price_rmse", "mean"),
            price_rmse_std=("price_rmse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mean_ratio_to_persistence=("price_rmse_ratio_to_persistence", "mean"),
        )
        .sort_values(["market", "feature_set", "horizon_step", "price_rmse_mean"])
    )


def _aggregate_folds(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    aggregations: dict[str, tuple[str, str]] = {"seeds_completed": ("seed", "nunique")}
    for metric in ["price_rmse", "price_mae", "rmse", "mae"]:
        if metric in frame:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_std"] = (metric, "std")
    return (
        frame.groupby(["market", "feature_set", "model", "fold"], as_index=False)
        .agg(**aggregations)
        .sort_values(["market", "feature_set", "fold", "price_rmse_mean", "model"])
    )


def _seed_stability(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby(["market", "feature_set", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            ratio_mean=("price_rmse_ratio_to_persistence", "mean"),
            ratio_std=("price_rmse_ratio_to_persistence", "std"),
            price_rmse_mean=("price_rmse", "mean"),
            price_rmse_std=("price_rmse", "std"),
        )
        .sort_values(["market", "feature_set", "ratio_mean", "model"])
    )


def _plot_model_rankings(path: Path, frame: pd.DataFrame) -> None:
    summary = frame.groupby("model")["price_rmse_rank"].agg(["mean", "std"]).sort_values("mean")
    _bar_plot(
        path,
        summary.index.tolist(),
        summary["mean"].to_numpy(),
        "Average model ranking across markets, features, and seeds",
        "Mean price RMSE rank (lower is better)",
        errors=summary["std"].fillna(0).to_numpy(),
    )


def _plot_indicator_effects(path: Path, paired: pd.DataFrame) -> None:
    if paired.empty:
        return
    summary = paired.groupby("model")["price_rmse_delta_pct"].agg(["mean", "std"])
    summary = summary.sort_values("mean")
    _bar_plot(
        path,
        summary.index.tolist(),
        summary["mean"].to_numpy(),
        "Effect of technical indicators",
        "Price RMSE change vs Close only (%)",
        errors=summary["std"].fillna(0).to_numpy(),
        horizontal_zero=True,
    )


def _plot_cross_market(path: Path, frame: pd.DataFrame) -> None:
    summary = frame.groupby(["market", "model"])["price_rmse_ratio_to_persistence"].mean()
    markets = sorted(frame["market"].unique())
    models = sorted(frame["model"].unique())
    values = np.asarray(
        [[summary.get((market, model), np.nan) for model in models] for market in markets]
    )
    _grouped_bar_plot(
        path,
        models,
        markets,
        values,
        "Cross-market performance",
        "Price RMSE / persistence RMSE",
        horizontal_one=True,
    )


def _plot_horizons(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6))
    summary = frame.groupby(["model", "horizon_step"])["price_rmse_ratio_to_persistence"].mean()
    for model in sorted(frame["model"].unique()):
        series = summary.loc[model].sort_index()
        axis.plot(series.index, series.values, marker="o", label=model)
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_title("Per-horizon performance across markets, features, and seeds")
    axis.set_xlabel("Recursive horizon step")
    axis.set_ylabel("Price RMSE / persistence RMSE")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_seed_stability(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6))
    summary = frame.groupby(["model", "seed"])["price_rmse_ratio_to_persistence"].mean()
    for model in sorted(frame["model"].unique()):
        series = summary.loc[model].sort_index()
        axis.plot(series.index.astype(str), series.values, marker="o", label=model)
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_title("Seed stability across markets and feature sets")
    axis.set_xlabel("Seed")
    axis.set_ylabel("Mean price RMSE / persistence RMSE")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _bar_plot(
    path: Path,
    labels: list[str],
    values: np.ndarray,
    title: str,
    ylabel: str,
    errors: np.ndarray | None = None,
    horizontal_zero: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, yerr=errors, capsize=3)
    if horizontal_zero:
        axis.axhline(0.0, color="black", linewidth=1)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _grouped_bar_plot(
    path: Path,
    categories: list[str],
    groups: list[str],
    values: np.ndarray,
    title: str,
    ylabel: str,
    horizontal_one: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.8 / max(len(groups), 1)
    for index, group in enumerate(groups):
        offset = (index - (len(groups) - 1) / 2) * width
        axis.bar(x + offset, values[index], width=width, label=group)
    if horizontal_one:
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(x, categories, rotation=30)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
