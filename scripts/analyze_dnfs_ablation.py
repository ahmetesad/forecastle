from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from forecastle.models import build_model
from forecastle.plotting import plt

SEEDS = [1, 7, 42, 123, 2026]
USAGE_BATCHES = {
    "0": Path("outputs/studies/dnfs_usage_ablation/dnfs_gru_first_order_usage_0"),
    "1e-4": Path("outputs/studies/dnfs_usage_coefficient_ablation/dnfs_usage_1e4"),
    "3e-4": Path("outputs/studies/dnfs_usage_coefficient_ablation/dnfs_usage_3e4"),
    "1e-3": Path("outputs/studies/dnfs_usage_ablation/dnfs_gru_first_order_usage_1e3"),
    "3e-3": Path("outputs/studies/dnfs_usage_coefficient_ablation/dnfs_usage_3e3"),
}
RULE_BATCHES = {
    "4": Path("outputs/studies/dnfs_rule_count_ablation/dnfs_rules4"),
    "8": USAGE_BATCHES["1e-3"],
    "16": Path("outputs/studies/dnfs_rule_count_ablation/dnfs_rules16"),
}
OUTPUT = Path("results/wig20/dnfs_ablation")


@dataclass(frozen=True)
class StudyFrames:
    seeds: pd.DataFrame
    folds: pd.DataFrame
    horizons: pd.DataFrame
    rules: pd.DataFrame
    origins: pd.DataFrame


def _write_table(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    (OUTPUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_study(batches: dict[str, Path], setting_column: str) -> StudyFrames:
    seed_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    horizon_frames: list[pd.DataFrame] = []
    rule_rows: list[dict[str, float | int | str]] = []
    origin_frames: list[pd.DataFrame] = []

    for setting, batch_dir in batches.items():
        results = pd.read_csv(batch_dir / "run_results.csv")
        models = results.loc[results["model"] == "dnfs"].copy()
        baselines = results.loc[results["model"] == "naive_persistence"].copy()
        if sorted(models["seed"].tolist()) != SEEDS:
            raise ValueError(f"Incomplete DNFS seeds for {setting}: {batch_dir}")
        models[setting_column] = setting
        models["improvement_vs_persistence_pct"] = (
            100
            * (models["persistence_price_rmse"] - models["price_rmse"])
            / models["persistence_price_rmse"]
        )
        seed_frames.append(models)
        baseline_artifacts = baselines.set_index("seed")["artifact_dir"].to_dict()

        for run in models.itertuples(index=False):
            artifact = Path(run.artifact_dir)
            baseline_artifact = Path(baseline_artifacts[run.seed])

            origins = pd.read_csv(artifact / "folds.csv")
            origins[setting_column] = setting
            origins["seed"] = run.seed
            origin_frames.append(origins)

            folds = pd.read_csv(artifact / "fold_metrics.csv").query("model == 'dnfs'")
            baseline_folds = (
                pd.read_csv(baseline_artifact / "fold_metrics.csv")
                .query("model == 'naive_persistence'")[["fold", "price_rmse"]]
                .rename(columns={"price_rmse": "persistence_price_rmse"})
            )
            folds = folds.merge(baseline_folds, on="fold", validate="one_to_one")
            folds[setting_column] = setting
            folds["seed"] = run.seed
            folds["improvement_vs_persistence_pct"] = (
                100
                * (folds["persistence_price_rmse"] - folds["price_rmse"])
                / folds["persistence_price_rmse"]
            )

            horizons = pd.read_csv(artifact / "horizon_metrics.csv").query("model == 'dnfs'")
            baseline_horizons = (
                pd.read_csv(baseline_artifact / "horizon_metrics.csv")
                .query("model == 'naive_persistence'")[["horizon_step", "price_rmse"]]
                .rename(columns={"price_rmse": "persistence_price_rmse"})
            )
            horizons = horizons.merge(baseline_horizons, on="horizon_step", validate="one_to_one")
            horizons[setting_column] = setting
            horizons["seed"] = run.seed
            horizons["improvement_vs_persistence_pct"] = (
                100
                * (horizons["persistence_price_rmse"] - horizons["price_rmse"])
                / horizons["persistence_price_rmse"]
            )

            activations = pd.read_csv(artifact / "rule_analysis" / "rule_activations.csv")
            sample_keys = [
                "fold",
                "forecast_origin",
                "target_date",
                "horizon_step",
            ]
            samples = activations.drop_duplicates(sample_keys)
            for fold, group in activations.groupby("fold"):
                usage = group.groupby("rule")["weight"].mean().to_numpy()
                entropy = float(-(usage * np.log(np.clip(usage, 1e-12, None))).sum())
                fold_samples = samples.loc[samples["fold"] == fold]
                sample_weights = group.pivot_table(
                    index=sample_keys, columns="rule", values="weight"
                )
                dominant_rule_fraction = float(
                    sample_weights.idxmax(axis=1).value_counts(normalize=True).max()
                )
                rule_rows.append(
                    {
                        setting_column: setting,
                        "seed": int(run.seed),
                        "fold": int(fold),
                        "effective_rules_per_fold": float(np.exp(entropy)),
                        "effective_rules_per_sample": float(
                            np.exp(fold_samples["rule_entropy"]).mean()
                        ),
                        "largest_mean_rule_usage": float(usage.max()),
                        "active_rules_above_1pct": int((usage >= 0.01).sum()),
                        "unused_rules_below_1pct": int((usage < 0.01).sum()),
                        "normalized_entropy": float(entropy / np.log(len(usage))),
                        "dominant_rule_fraction": dominant_rule_fraction,
                    }
                )
            fold_frames.append(folds)
            horizon_frames.append(horizons)

    return StudyFrames(
        seeds=pd.concat(seed_frames, ignore_index=True),
        folds=pd.concat(fold_frames, ignore_index=True),
        horizons=pd.concat(horizon_frames, ignore_index=True),
        rules=pd.DataFrame(rule_rows),
        origins=pd.concat(origin_frames, ignore_index=True),
    )


def _performance_summary(frames: StudyFrames, setting_column: str) -> pd.DataFrame:
    summary = frames.seeds.groupby(setting_column, as_index=False).agg(
        seeds=("seed", "nunique"),
        price_rmse_mean=("price_rmse", "mean"),
        price_rmse_std=("price_rmse", "std"),
        price_mae_mean=("price_mae", "mean"),
        return_rmse_mean=("rmse", "mean"),
        return_mae_mean=("mae", "mean"),
        improvement_vs_persistence_pct_mean=(
            "improvement_vs_persistence_pct",
            "mean",
        ),
        training_time_seconds_mean=("training_time_seconds", "mean"),
        inference_time_seconds_mean=("inference_time_seconds", "mean"),
    )
    seed_wins = frames.seeds.groupby(setting_column)["improvement_vs_persistence_pct"].apply(
        lambda values: int((values > 0).sum())
    )
    fold_wins = frames.folds.groupby(setting_column)["improvement_vs_persistence_pct"].agg(
        [lambda values: int((values > 0).sum()), "count"]
    )
    horizon_wins = frames.horizons.groupby(setting_column)["improvement_vs_persistence_pct"].agg(
        [lambda values: int((values > 0).sum()), "count"]
    )
    summary["seeds_won_vs_persistence"] = summary[setting_column].map(seed_wins)
    summary["fold_win_rate"] = summary[setting_column].map(
        fold_wins["<lambda_0>"] / fold_wins["count"]
    )
    summary["horizon_step_win_rate"] = summary[setting_column].map(
        horizon_wins["<lambda_0>"] / horizon_wins["count"]
    )
    return summary


def _rule_summary(frames: StudyFrames, setting_column: str) -> pd.DataFrame:
    return frames.rules.groupby(setting_column, as_index=False).agg(
        effective_rules_per_fold=("effective_rules_per_fold", "mean"),
        effective_rules_per_sample=("effective_rules_per_sample", "mean"),
        largest_mean_rule_usage=("largest_mean_rule_usage", "mean"),
        active_rules_above_1pct=("active_rules_above_1pct", "mean"),
        unused_rules_below_1pct=("unused_rules_below_1pct", "mean"),
        normalized_entropy=("normalized_entropy", "mean"),
        dominant_rule_fraction=("dominant_rule_fraction", "mean"),
    )


def _balance_accuracy_summary(frames: StudyFrames, setting_column: str) -> pd.DataFrame:
    merged = frames.folds.merge(
        frames.rules,
        on=[setting_column, "seed", "fold"],
        validate="one_to_one",
    )
    rows = []
    for setting, group in merged.groupby(setting_column):
        rows.append(
            {
                setting_column: setting,
                "effective_rules_accuracy_correlation": group["effective_rules_per_fold"].corr(
                    group["improvement_vs_persistence_pct"]
                ),
                "entropy_accuracy_correlation": group["normalized_entropy"].corr(
                    group["improvement_vs_persistence_pct"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_seed_differences(
    frames: StudyFrames, setting_column: str, references: list[str]
) -> pd.DataFrame:
    pivot = frames.seeds.pivot(index="seed", columns=setting_column, values="price_rmse")
    rows: list[dict[str, float | int | str]] = []
    for setting in pivot.columns:
        for seed in pivot.index:
            row: dict[str, float | int | str] = {
                setting_column: str(setting),
                "seed": int(seed),
                "price_rmse": float(pivot.loc[seed, setting]),
            }
            for reference in references:
                row[f"price_rmse_delta_vs_{reference}"] = float(
                    pivot.loc[seed, setting] - pivot.loc[seed, reference]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _horizon_summary(frames: StudyFrames, setting_column: str) -> pd.DataFrame:
    return frames.horizons.groupby([setting_column, "horizon_step"], as_index=False).agg(
        price_rmse_mean=("price_rmse", "mean"),
        price_rmse_std=("price_rmse", "std"),
        persistence_price_rmse=("persistence_price_rmse", "mean"),
        improvement_vs_persistence_pct_mean=(
            "improvement_vs_persistence_pct",
            "mean",
        ),
        seeds_won_vs_persistence=(
            "improvement_vs_persistence_pct",
            lambda values: int((values > 0).sum()),
        ),
    )


def _fold_summary(frames: StudyFrames, setting_column: str) -> pd.DataFrame:
    return frames.folds.groupby([setting_column, "fold"], as_index=False).agg(
        price_rmse_mean=("price_rmse", "mean"),
        price_rmse_std=("price_rmse", "std"),
        improvement_vs_persistence_pct_mean=(
            "improvement_vs_persistence_pct",
            "mean",
        ),
        seeds_won_vs_persistence=(
            "improvement_vs_persistence_pct",
            lambda values: int((values > 0).sum()),
        ),
    )


def _verify_design(*studies: StudyFrames) -> pd.DataFrame:
    origin_sets = []
    persistence_values = []
    for frames in studies:
        origin_sets.extend(
            tuple(group.sort_values("fold")["forecast_origin"])
            for _, group in frames.origins.groupby(
                [
                    column
                    for column in frames.origins.columns
                    if column in {"seed", "usage_regularization", "num_rules"}
                ]
            )
        )
        persistence_values.extend(frames.seeds["persistence_price_rmse"].tolist())
    return pd.DataFrame(
        [
            {
                "check": "identical_forecast_origins",
                "passed": len(set(origin_sets)) == 1,
                "observed_unique_values": len(set(origin_sets)),
            },
            {
                "check": "identical_persistence_price_rmse",
                "passed": len({round(value, 12) for value in persistence_values}) == 1,
                "observed_unique_values": len({round(value, 12) for value in persistence_values}),
            },
        ]
    )


def _parameter_count(num_rules: int) -> int:
    model = build_model(
        "dnfs",
        sequence_length=30,
        feature_count=9,
        params={
            "encoder_type": "gru",
            "encoder_hidden_size": 64,
            "encoder_num_layers": 1,
            "latent_size": 32,
            "num_rules": num_rules,
            "consequent_type": "first_order",
        },
    )
    return sum(parameter.numel() for parameter in model.parameters())


def _plots(usage: StudyFrames, rules: StudyFrames) -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    for setting, group in usage.seeds.groupby("usage_regularization"):
        axis.plot(group["seed"].astype(str), group["price_rmse"], marker="o", label=setting)
    axis.axhline(
        usage.seeds["persistence_price_rmse"].iloc[0],
        color="black",
        linestyle="--",
        label="persistence",
    )
    axis.set(xlabel="Seed", ylabel="Price RMSE", title="DNFS seed stability")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "phase1_seed_rmse.png", dpi=150)
    plt.close(figure)

    aggregate = _performance_summary(usage, "usage_regularization")
    numeric_order = {"0": 0.0, "1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3}
    aggregate["coefficient"] = aggregate["usage_regularization"].map(numeric_order)
    aggregate = aggregate.sort_values("coefficient")
    coefficient_positions = np.arange(len(aggregate))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        coefficient_positions,
        aggregate["price_rmse_mean"],
        yerr=aggregate["price_rmse_std"],
        marker="o",
        capsize=4,
    )
    axis.axhline(usage.seeds["persistence_price_rmse"].iloc[0], color="black", linestyle="--")
    axis.set_xticks(coefficient_positions, aggregate["usage_regularization"])
    axis.set(xlabel="Usage regularization", ylabel="Price RMSE", title="Coefficient sensitivity")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "phase1_coefficient_rmse.png", dpi=150)
    plt.close(figure)

    horizon = _horizon_summary(usage, "usage_regularization")
    figure, axis = plt.subplots(figsize=(10, 5))
    for setting, group in horizon.groupby("usage_regularization"):
        axis.plot(
            group["horizon_step"], group["improvement_vs_persistence_pct_mean"], label=setting
        )
    axis.axhline(0, color="black", linestyle="--")
    axis.set(
        xlabel="Recursive horizon step",
        ylabel="Improvement over persistence (%)",
        title="Phase 1 horizon performance",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "phase1_horizon_improvement.png", dpi=150)
    plt.close(figure)

    merged = usage.folds.merge(
        usage.rules,
        on=["usage_regularization", "seed", "fold"],
        validate="one_to_one",
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    for setting, group in merged.groupby("usage_regularization"):
        axis.scatter(
            group["effective_rules_per_fold"],
            group["improvement_vs_persistence_pct"],
            alpha=0.65,
            label=setting,
        )
    axis.axhline(0, color="black", linestyle="--")
    axis.set(
        xlabel="Effective rules per fold",
        ylabel="Improvement over persistence (%)",
        title="Rule balance versus accuracy",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "phase1_balance_vs_accuracy.png", dpi=150)
    plt.close(figure)

    rule_aggregate = _performance_summary(rules, "num_rules")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        rule_aggregate["num_rules"].astype(int),
        rule_aggregate["price_rmse_mean"],
        yerr=rule_aggregate["price_rmse_std"],
        marker="o",
        capsize=4,
    )
    axis.axhline(rules.seeds["persistence_price_rmse"].iloc[0], color="black", linestyle="--")
    axis.set(xlabel="Configured rules", ylabel="Price RMSE", title="Rule-count sensitivity")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "phase2_rule_count_rmse.png", dpi=150)
    plt.close(figure)

    rule_usage = _rule_summary(rules, "num_rules")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        rule_usage["num_rules"].astype(int),
        rule_usage["effective_rules_per_fold"],
        marker="o",
        label="effective",
    )
    axis.plot(
        rule_usage["num_rules"].astype(int),
        rule_usage["active_rules_above_1pct"],
        marker="o",
        label="active >1%",
    )
    axis.set(xlabel="Configured rules", ylabel="Mean rules", title="Rule utilization")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "phase2_rule_utilization.png", dpi=150)
    plt.close(figure)

    usage_rule_summary = _rule_summary(usage, "usage_regularization")
    usage_rule_summary["coefficient"] = usage_rule_summary["usage_regularization"].map(
        numeric_order
    )
    usage_rule_summary = usage_rule_summary.sort_values("coefficient")
    rule_positions = np.arange(len(usage_rule_summary))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        rule_positions,
        usage_rule_summary["effective_rules_per_fold"],
        marker="o",
        label="effective rules",
    )
    axis.plot(
        rule_positions,
        usage_rule_summary["active_rules_above_1pct"],
        marker="o",
        label="active >1%",
    )
    axis.set_xticks(rule_positions, usage_rule_summary["usage_regularization"])
    axis.set(
        xlabel="Usage regularization",
        ylabel="Mean rules",
        title="Phase 1 rule utilization",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "phase1_rule_utilization.png", dpi=150)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    usage = _load_study(USAGE_BATCHES, "usage_regularization")
    rules = _load_study(RULE_BATCHES, "num_rules")

    usage_performance = _performance_summary(usage, "usage_regularization")
    usage_rules = _rule_summary(usage, "usage_regularization")
    usage_aggregate = usage_performance.merge(
        usage_rules, on="usage_regularization", validate="one_to_one"
    )
    usage_order = {"0": 0, "1e-4": 1, "3e-4": 2, "1e-3": 3, "3e-3": 4}
    usage_aggregate = usage_aggregate.sort_values(
        "usage_regularization", key=lambda values: values.map(usage_order)
    )
    _write_table(usage_aggregate, "phase1_usage_coefficients")
    _write_table(
        _paired_seed_differences(usage, "usage_regularization", ["0", "1e-3"]),
        "phase1_paired_seed_differences",
    )
    _write_table(_fold_summary(usage, "usage_regularization"), "phase1_fold_results")
    _write_table(_horizon_summary(usage, "usage_regularization"), "phase1_horizon_results")
    _write_table(usage.rules, "phase1_rule_usage_by_fold")
    _write_table(
        _balance_accuracy_summary(usage, "usage_regularization"),
        "phase1_balance_accuracy_correlations",
    )

    rule_performance = _performance_summary(rules, "num_rules")
    rule_usage = _rule_summary(rules, "num_rules")
    rule_aggregate = rule_performance.merge(rule_usage, on="num_rules", validate="one_to_one")
    rule_aggregate["parameter_count"] = rule_aggregate["num_rules"].map(
        lambda value: _parameter_count(int(value))
    )
    rule_aggregate["effective_rules_to_total_ratio"] = rule_aggregate[
        "effective_rules_per_fold"
    ] / rule_aggregate["num_rules"].astype(int)
    rule_aggregate = rule_aggregate.sort_values("num_rules", key=lambda values: values.astype(int))
    _write_table(rule_aggregate, "phase2_rule_counts")
    _write_table(
        _paired_seed_differences(rules, "num_rules", ["8"]),
        "phase2_paired_seed_differences",
    )
    _write_table(_fold_summary(rules, "num_rules"), "phase2_fold_results")
    _write_table(_horizon_summary(rules, "num_rules"), "phase2_horizon_results")
    _write_table(rules.rules, "phase2_rule_usage_by_fold")
    _write_table(_verify_design(usage, rules), "design_verification")
    _plots(usage, rules)


if __name__ == "__main__":
    main()
