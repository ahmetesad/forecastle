from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forecastle.artifacts import write_dataframe
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
    manifest, results, horizons = _collect_completed_results(batch_dir)
    write_dataframe(batch_dir / "manifest.csv", manifest)
    write_dataframe(batch_dir / "run_results.csv", results)
    write_dataframe(batch_dir / "horizon_results.csv", horizons)
    if results.empty:
        return

    results = _add_persistence_comparisons(results)
    horizons = _add_horizon_persistence_comparisons(horizons)
    write_dataframe(batch_dir / "run_results.csv", results)
    write_dataframe(batch_dir / "horizon_results.csv", horizons)

    aggregate = _aggregate_metrics(results)
    rankings = _aggregate_rankings(results)
    indicator_pairs, indicator_summary = _indicator_effects(results)
    horizon_summary = _aggregate_horizons(horizons)
    seed_stability = _seed_stability(results)

    write_dataframe(batch_dir / "aggregate_metrics.csv", aggregate)
    write_dataframe(batch_dir / "model_rankings.csv", rankings)
    write_dataframe(batch_dir / "indicator_effects.csv", indicator_pairs)
    write_dataframe(batch_dir / "indicator_effect_summary.csv", indicator_summary)
    write_dataframe(batch_dir / "cross_market_comparison.csv", aggregate)
    write_dataframe(batch_dir / "aggregate_horizon_metrics.csv", horizon_summary)
    write_dataframe(batch_dir / "seed_stability.csv", seed_stability)

    plots_dir = batch_dir / "plots"
    _plot_model_rankings(plots_dir / "model_rankings.png", results)
    _plot_indicator_effects(plots_dir / "indicator_effects.png", indicator_pairs)
    _plot_cross_market(plots_dir / "cross_market_comparison.png", results)
    _plot_horizons(plots_dir / "per_horizon_performance.png", horizons)
    _plot_seed_stability(plots_dir / "seed_stability.png", results)


def validate_run_artifacts(artifact_dir: Path, expected_model: str) -> None:
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


def _collect_completed_results(batch_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_rows = []
    result_frames = []
    horizon_frames = []
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
    manifest = pd.DataFrame(manifest_rows)
    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    horizons = pd.concat(horizon_frames, ignore_index=True) if horizon_frames else pd.DataFrame()
    return manifest, results, horizons


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


def _aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "runs": ("run_id", "count"),
        "seeds_completed": ("seed", "nunique"),
    }
    for metric in SUMMARY_METRICS:
        if metric in frame:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_std"] = (metric, "std")
    aggregate = (
        frame.groupby(["market", "feature_set", "model"], as_index=False)
        .agg(**aggregations)
        .sort_values(["market", "feature_set", "price_rmse_mean", "model"])
    )
    aggregate["price_rmse_rank"] = aggregate.groupby(["market", "feature_set"])[
        "price_rmse_mean"
    ].rank(method="min")
    return aggregate


def _aggregate_rankings(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["feature_set", "model"], as_index=False)
        .agg(
            mean_price_rmse_rank=("price_rmse_rank", "mean"),
            std_price_rmse_rank=("price_rmse_rank", "std"),
            mean_ratio_to_persistence=("price_rmse_ratio_to_persistence", "mean"),
            runs=("run_id", "count"),
        )
        .sort_values(["feature_set", "mean_price_rmse_rank", "model"])
    )


def _indicator_effects(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _seed_stability(frame: pd.DataFrame) -> pd.DataFrame:
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
