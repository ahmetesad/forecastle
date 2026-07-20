from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
import yaml

from forecastle.comparison import BatchComparisonError, run_comparison

if TYPE_CHECKING:
    from pathlib import Path

MODELS = ["naive_persistence", "cnn1d"]


def test_direct_comparison_pairs_only_endpoint_and_checks_persistence(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_batch(reference, recursive=True)
    _write_batch(candidate, recursive=False, target_metric_delta=5e-9)
    config_path = _write_config(
        tmp_path,
        "direct_vs_recursive",
        reference,
        candidate,
    )

    output_dir = run_comparison(config_path)

    paired = pd.read_csv(output_dir / "paired_results.csv")
    horizons = pd.read_csv(output_dir / "horizon_comparison.csv")
    integrity = pd.read_csv(output_dir / "schedule_integrity.csv")
    assert len(paired) == 2
    assert horizons["horizon_step"].unique().tolist() == [20]
    assert integrity["integrity_pass"].astype(bool).all()
    cnn = paired[paired["model"].eq("cnn1d")].iloc[0]
    assert cnn["price_rmse_delta"] == pytest.approx(-1.0)
    assert (output_dir / "comparison_summary.md").exists()
    assert (output_dir / "plots/model_price_rmse_delta.png").exists()


def test_comparison_rejects_persistence_metric_mismatch(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_batch(reference, recursive=True)
    _write_batch(candidate, recursive=False, persistence_rmse=11.0)
    config_path = _write_config(
        tmp_path,
        "direct_vs_recursive",
        reference,
        candidate,
    )

    with pytest.raises(BatchComparisonError, match="persistence differs"):
        run_comparison(config_path)


def test_rolling_comparison_writes_fold_and_horizon_reports(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_batch(reference, recursive=True)
    _write_batch(candidate, recursive=True)
    config_path = _write_config(
        tmp_path,
        "rolling_vs_expanding",
        reference,
        candidate,
    )

    output_dir = run_comparison(config_path)

    assert (output_dir / "fold_comparison.csv").exists()
    assert (output_dir / "fold_comparison_summary.csv").exists()
    horizons = pd.read_csv(output_dir / "horizon_comparison.csv")
    assert sorted(horizons["horizon_step"].unique()) == [1, 20]


def _write_batch(
    root: Path,
    *,
    recursive: bool,
    persistence_rmse: float = 10.0,
    target_metric_delta: float = 0.0,
) -> None:
    root.mkdir(parents=True)
    identities = [
        {
            "run_id": f"wig20__{model}__close__seed42",
            "market": "wig20",
            "model": model,
            "feature_set": "close",
            "seed": 42,
            "status": "completed",
        }
        for model in MODELS
    ]
    pd.DataFrame(identities).to_csv(root / "manifest.csv", index=False)
    run_rows = []
    horizon_rows = []
    fold_rows = []
    for model in MODELS:
        price_rmse = persistence_rmse if model == "naive_persistence" else 8.0
        if not recursive and model == "cnn1d":
            price_rmse = 7.0
        metrics = {
            "market": "wig20",
            "model": model,
            "feature_set": "close",
            "seed": 42,
            "price_rmse": price_rmse,
            "price_mae": price_rmse - 1.0,
            "rmse": price_rmse / 100.0,
            "mae": (price_rmse - 1.0) / 100.0,
        }
        if model == "naive_persistence":
            metrics["rmse"] += target_metric_delta
            metrics["mae"] += target_metric_delta
        run_rows.append(metrics)
        steps = [1, 20] if recursive else [20]
        for step in steps:
            horizon_rows.append({**metrics, "horizon_step": step})
        fold_rows.append({**metrics, "fold": 0})
    pd.DataFrame(run_rows).to_csv(root / "run_results.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(root / "horizon_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(root / "fold_results.csv", index=False)

    plan_rows = []
    steps = [1, 20] if recursive else [20]
    for step in steps:
        plan_rows.append(
            {
                "fold": 0,
                "forecast_origin": "2024-01-01T00:00:00",
                "target_date": ("2024-01-02T00:00:00" if step == 1 else "2024-01-29T00:00:00"),
                "horizon_step": step,
            }
        )
    plans = root / "matched_origins"
    plans.mkdir()
    pd.DataFrame(plan_rows).to_csv(plans / "wig20_plan.csv", index=False)


def _write_config(
    tmp_path: Path,
    kind: str,
    reference: Path,
    candidate: Path,
) -> Path:
    path = tmp_path / f"{kind}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "comparison": {
                    "name": kind,
                    "kind": kind,
                    "horizon_step": 20,
                    "output_dir": str(tmp_path / "comparisons"),
                    "reference": {
                        "label": "reference",
                        "batch_dir": str(reference),
                    },
                    "candidate": {
                        "label": "candidate",
                        "batch_dir": str(candidate),
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
