from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import forecastle.batch as batch_module
from forecastle.batch import expand_batch_runs, run_batch
from forecastle.batch_integrity import write_matched_origin_integrity_report
from forecastle.evaluation.errors import RecursiveForecastDivergence
from forecastle.evaluation.matched import MatchedOriginIntegrityError


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_market_batch_expands_to_stable_single_model_runs(tmp_path) -> None:
    batch_raw = load_yaml(Path("configs/batches/markets_matched_origins_recursive_h20.yaml"))
    base_raw = load_yaml(Path(batch_raw["base_config"]))

    runs = expand_batch_runs(base_raw, batch_raw["batch"], tmp_path)

    assert len(runs) == 180
    assert len({run.run_id for run in runs}) == 180
    selected = next(run for run in runs if run.run_id == "wig20__cnn1d__close__seed42")
    assert [model.name for model in selected.config.training.models] == ["cnn1d"]
    assert selected.config.training.baselines == []
    assert selected.config.dataset.technical_indicators is None
    assert selected.config.dataset.horizon == 20
    assert selected.config.forecasting.strategy == "recursive"
    assert selected.config.evaluation.strategy == "walk_forward"
    assert selected.config.dataset.aligned_warmup_rows == 33
    assert selected.config.evaluation.matched_plan_path is not None

    baseline = next(
        run for run in runs if run.run_id == "sp500__naive_persistence__indicators__seed2026"
    )
    assert baseline.config.training.models == []
    assert baseline.config.training.baselines == ["naive_persistence"]
    assert baseline.config.dataset.technical_indicators is not None

    dnfs = next(run for run in runs if run.run_id == "bist100__dnfs__indicators__seed42")
    assert dnfs.config.training.models[0].params == {
        "encoder_type": "gru",
        "consequent_type": "first_order",
        "num_rules": 8,
        "usage_regularization": 0.001,
        "rule_initialization": "kmeans",
    }


def test_batch_resumes_and_writes_aggregate_artifacts(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "prices.csv"
    values = np.linspace(100.0, 130.0, 120)
    pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=len(values)),
            "Close": values,
        }
    ).to_csv(csv_path, index=False)

    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "output_dir": str(tmp_path / "unused")},
                "dataset": {
                    "name": "synthetic",
                    "csv_path": str(csv_path),
                    "date_column": "Date",
                    "target_column": "Close",
                    "feature_columns": ["Close"],
                    "target_transform": "log_return",
                    "sequence_length": 5,
                    "horizon": 2,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                },
                "forecasting": {"strategy": "recursive"},
                "evaluation": {
                    "strategy": "walk_forward",
                    "window": "expanding",
                    "max_folds": 1,
                },
                "training": {
                    "batch_size": 8,
                    "epochs": 1,
                    "patience": 1,
                    "models": [{"name": "mlp", "params": {"hidden_sizes": [4]}}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base_path),
                "batch": {
                    "name": "resume_test",
                    "output_dir": str(tmp_path / "batches"),
                    "matched_origins": True,
                    "datasets": [
                        {
                            "name": "synthetic",
                            "csv_path": str(csv_path),
                            "date_column": "Date",
                            "target_column": "Close",
                        }
                    ],
                    "models": ["naive_persistence"],
                    "feature_sets": {
                        "close": {
                            "feature_columns": ["Close"],
                            "technical_indicators": None,
                        },
                        "indicators": {
                            "feature_columns": ["Close"],
                            "technical_indicators": {
                                "sma_periods": [3],
                                "rsi_period": 3,
                                "macd": {
                                    "fast_period": 2,
                                    "slow_period": 4,
                                    "signal_period": 2,
                                },
                            },
                        },
                    },
                    "seeds": [7],
                    "horizon": 2,
                    "forecasting": {"strategy": "recursive"},
                    "evaluation": {
                        "strategy": "walk_forward",
                        "window": "expanding",
                        "max_folds": 1,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    calls = 0
    real_run_experiment = batch_module.run_experiment

    def counted_run(config):
        nonlocal calls
        calls += 1
        return real_run_experiment(config)

    monkeypatch.setattr(batch_module, "run_experiment", counted_run)
    batch_dir = run_batch(config_path)
    assert calls == 2

    resumed_dir = run_batch(config_path)
    assert resumed_dir == batch_dir
    assert calls == 2
    close_root = batch_dir / "runs/synthetic__naive_persistence__close__seed7"
    metadata_path = close_root / "metadata.yaml"
    metadata = load_yaml(metadata_path)
    metadata["status"] = "interrupted"
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    run_batch(config_path)
    assert calls == 3
    assert (close_root / "config.yaml").exists()
    assert load_yaml(metadata_path)["status"] == "completed"
    for filename in [
        "aggregate_metrics.csv",
        "model_rankings.csv",
        "indicator_effects.csv",
        "cross_market_comparison.csv",
        "aggregate_horizon_metrics.csv",
        "aggregate_fold_metrics.csv",
        "seed_stability.csv",
        "completion_summary.csv",
        "divergence_frequency.csv",
        "matched_origin_integrity.csv",
    ]:
        assert (batch_dir / filename).exists()
        assert (batch_dir / filename).with_suffix(".md").exists()
    for filename in [
        "model_rankings.png",
        "indicator_effects.png",
        "cross_market_comparison.png",
        "per_horizon_performance.png",
        "seed_stability.png",
    ]:
        assert (batch_dir / "plots" / filename).exists()

    integrity = pd.read_csv(batch_dir / "matched_origin_integrity.csv")
    checked = integrity[integrity["checked"].astype(bool)]
    assert not checked.empty
    assert checked["integrity_pass"].astype(bool).all()


def test_matched_integrity_rejects_changed_persistence_prediction(tmp_path, monkeypatch) -> None:
    original = batch_module.write_matched_origin_integrity_report

    def skip_during_batch(batch_dir):
        return original(batch_dir, raise_on_mismatch=False)

    monkeypatch.setattr(batch_module, "write_matched_origin_integrity_report", skip_during_batch)
    batch_dir = _run_small_matched_persistence_batch(tmp_path)
    prediction_path = next(
        (batch_dir / "runs/synthetic__naive_persistence__indicators__seed7/artifacts").glob(
            "*/predictions/naive_persistence_predictions.csv"
        )
    )
    predictions = pd.read_csv(prediction_path)
    predictions.loc[0, "prediction_price"] += 1.0
    predictions.to_csv(prediction_path, index=False)

    with pytest.raises(MatchedOriginIntegrityError, match="Persistence predictions"):
        write_matched_origin_integrity_report(batch_dir)


def test_batch_records_structured_recursive_divergence(tmp_path, monkeypatch) -> None:
    config_path = _write_single_run_batch(tmp_path)

    def diverge(_config):
        raise RecursiveForecastDivergence(
            model="linear_regression",
            fold=4,
            forecast_origin="2025-01-02T00:00:00",
            horizon_step=12,
            previous_price=123.0,
            predicted_target=1_000.0,
            reconstructed_price=float("inf"),
        )

    monkeypatch.setattr(batch_module, "run_experiment", diverge)
    batch_dir = run_batch(config_path)
    metadata = load_yaml(
        batch_dir / "runs/synthetic__linear_regression__close__seed7/metadata.yaml"
    )

    assert metadata["status"] == "failed"
    assert metadata["divergence"] is True
    assert metadata["divergence_fold"] == 4
    assert metadata["divergence_horizon_step"] == 12
    divergence = pd.read_csv(batch_dir / "divergence_frequency.csv")
    assert divergence.loc[0, "divergence_count"] == 1
    aggregate = pd.read_csv(batch_dir / "aggregate_metrics.csv")
    assert aggregate.loc[0, "completion_state"] == "failed"
    assert not bool(aggregate.loc[0, "eligible_for_ranking"])
    rankings = pd.read_csv(batch_dir / "model_rankings.csv")
    assert rankings.loc[0, "failed_combinations"] == 1


def test_batch_preserves_unchanged_failures_unless_retry_is_requested(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = _write_single_run_batch(tmp_path)
    calls = 0

    def diverge(_config):
        nonlocal calls
        calls += 1
        raise RecursiveForecastDivergence(
            model="linear_regression",
            fold=4,
            forecast_origin="2025-01-02T00:00:00",
            horizon_step=12,
            previous_price=123.0,
            predicted_target=1_000.0,
            reconstructed_price=float("inf"),
        )

    monkeypatch.setattr(batch_module, "_git_revision", lambda: "fixed-revision")
    monkeypatch.setattr(batch_module, "run_experiment", diverge)
    batch_dir = run_batch(config_path)
    run_batch(config_path)
    assert calls == 1
    metadata_path = batch_dir / "runs/synthetic__linear_regression__close__seed7/metadata.yaml"
    assert load_yaml(metadata_path)["last_action"] == "skipped_failed"

    run_batch(config_path, retry_failed=True)
    assert calls == 2
    assert load_yaml(metadata_path)["last_action"] == "failed"


def test_matched_batch_dry_run_materializes_matrix_without_experiments(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = _write_single_run_batch(tmp_path, matched=True, include_indicators=True)

    def unexpected_run(_config):
        raise AssertionError("dry-run must not launch experiments")

    monkeypatch.setattr(batch_module, "run_experiment", unexpected_run)
    batch_dir = run_batch(config_path, dry_run=True)

    planned = pd.read_csv(batch_dir / "planned_runs.csv")
    manifest = pd.read_csv(batch_dir / "manifest.csv")
    assert len(planned) == 2
    assert set(manifest["status"]) == {"planned"}
    assert (batch_dir / "matched_origins/synthetic_plan.csv").exists()
    assert not (batch_dir / "runs").exists()


def test_direct_and_rolling_batches_reuse_canonical_forecast_schedule(tmp_path) -> None:
    canonical_config = _write_single_run_batch(
        tmp_path,
        matched=True,
        include_indicators=True,
        name="canonical",
        max_folds=3,
    )
    canonical_dir = run_batch(canonical_config, dry_run=True)
    source_path = canonical_dir / "matched_origins/synthetic_plan.csv"
    canonical_plan = pd.read_csv(source_path)

    direct_config = _write_single_run_batch(
        tmp_path,
        matched=True,
        include_indicators=True,
        name="direct",
        strategy="direct",
        max_folds=3,
        origin_schedule_source=source_path,
    )
    direct_dir = run_batch(direct_config, dry_run=True)
    direct_plan = pd.read_csv(direct_dir / "matched_origins/synthetic_plan.csv")
    assert direct_plan["horizon_step"].unique().tolist() == [2]
    expected_direct = canonical_plan[canonical_plan["horizon_step"].eq(2)][
        ["fold", "forecast_origin", "target_date", "horizon_step"]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        direct_plan[["fold", "forecast_origin", "target_date", "horizon_step"]],
        expected_direct,
    )

    rolling_direct_config = _write_single_run_batch(
        tmp_path,
        matched=True,
        include_indicators=True,
        name="rolling_direct",
        strategy="direct",
        window="rolling",
        max_folds=3,
        origin_schedule_source=direct_dir / "matched_origins/synthetic_plan.csv",
    )
    rolling_direct_dir = run_batch(rolling_direct_config, dry_run=True)
    rolling_direct_plan = pd.read_csv(rolling_direct_dir / "matched_origins/synthetic_plan.csv")
    pd.testing.assert_frame_equal(
        rolling_direct_plan[["fold", "forecast_origin", "target_date", "horizon_step"]],
        expected_direct,
    )
    assert rolling_direct_plan.groupby("fold")["train_samples"].first().nunique() == 1

    rolling_config = _write_single_run_batch(
        tmp_path,
        matched=True,
        include_indicators=True,
        name="rolling",
        window="rolling",
        max_folds=3,
        origin_schedule_source=source_path,
    )
    rolling_dir = run_batch(rolling_config, dry_run=True)
    rolling_plan = pd.read_csv(rolling_dir / "matched_origins/synthetic_plan.csv")
    pd.testing.assert_frame_equal(
        rolling_plan[["fold", "forecast_origin", "target_date", "horizon_step"]],
        canonical_plan[["fold", "forecast_origin", "target_date", "horizon_step"]],
    )
    assert rolling_plan.groupby("fold")["train_samples"].first().nunique() == 1
    assert canonical_plan.groupby("fold")["train_samples"].first().nunique() > 1


def _run_small_matched_persistence_batch(tmp_path) -> Path:
    return run_batch(_write_single_run_batch(tmp_path, matched=True, include_indicators=True))


def _write_single_run_batch(
    tmp_path,
    *,
    matched: bool = False,
    include_indicators: bool = False,
    name: str = "helper_batch",
    strategy: str = "recursive",
    window: str = "expanding",
    max_folds: int = 1,
    origin_schedule_source: Path | None = None,
) -> Path:
    csv_path = tmp_path / "helper_prices.csv"
    values = np.linspace(100.0, 130.0, 120)
    pd.DataFrame(
        {"Date": pd.date_range("2020-01-01", periods=len(values)), "Close": values}
    ).to_csv(csv_path, index=False)
    base_path = tmp_path / "helper_base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "output_dir": str(tmp_path / "unused")},
                "dataset": {
                    "name": "synthetic",
                    "csv_path": str(csv_path),
                    "date_column": "Date",
                    "target_column": "Close",
                    "feature_columns": ["Close"],
                    "target_transform": "log_return",
                    "sequence_length": 5,
                    "horizon": 2,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                },
                "forecasting": {"strategy": "recursive"},
                "evaluation": {"strategy": "walk_forward", "max_folds": 1},
                "training": {
                    "batch_size": 8,
                    "epochs": 1,
                    "patience": 1,
                    "models": [{"name": "mlp", "params": {"hidden_sizes": [4]}}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    feature_sets = {"close": {"feature_columns": ["Close"], "technical_indicators": None}}
    if include_indicators:
        feature_sets["indicators"] = {
            "feature_columns": ["Close"],
            "technical_indicators": {
                "sma_periods": [3],
                "rsi_period": 3,
                "macd": {"fast_period": 2, "slow_period": 4, "signal_period": 2},
            },
        }
    batch_payload = {
        "name": name,
        "output_dir": str(tmp_path / "helper_batches"),
        "matched_origins": matched,
        "datasets": [
            {
                "name": "synthetic",
                "csv_path": str(csv_path),
                "date_column": "Date",
                "target_column": "Close",
            }
        ],
        "models": ["naive_persistence" if matched else "linear_regression"],
        "feature_sets": feature_sets,
        "seeds": [7],
        "horizon": 2,
        "forecasting": {"strategy": strategy},
        "evaluation": {
            "strategy": "walk_forward",
            "window": window,
            "max_folds": max_folds,
        },
    }
    if origin_schedule_source is not None:
        batch_payload["origin_schedule_sources"] = {"synthetic": str(origin_schedule_source)}
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base_path),
                "batch": batch_payload,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
