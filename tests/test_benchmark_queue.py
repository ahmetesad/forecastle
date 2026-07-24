from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from forecastle.batch import expand_batch_runs, run_batch
from forecastle.benchmark_queue import (
    BenchmarkClaimError,
    _benchmark_claim,
    normalize_devices,
    prepare_benchmark_specs,
    run_benchmark_queue,
)


def test_device_normalization_accepts_physical_cuda_ids() -> None:
    assert normalize_devices(["0", "cuda:1"]) == ["cuda:0", "cuda:1"]

    with pytest.raises(ValueError, match="duplicates"):
        normalize_devices(["cuda:0", "0"])
    with pytest.raises(ValueError, match="physical CUDA"):
        normalize_devices(["auto"])


def test_queue_rejects_shared_benchmark_output_directory(tmp_path) -> None:
    first = _write_benchmark(tmp_path, "first", batch_name="shared")
    second = _write_benchmark(tmp_path, "second", batch_name="shared")

    with pytest.raises(ValueError, match="separate output directories"):
        prepare_benchmark_specs([first, second])


def test_dry_run_writes_dynamic_schedule_without_training(tmp_path) -> None:
    configs = [
        _write_benchmark(tmp_path, "canada"),
        _write_benchmark(tmp_path, "france"),
        _write_benchmark(tmp_path, "germany"),
    ]

    queue_dir = run_benchmark_queue(
        configs,
        ["cuda:0", "cuda:1"],
        dry_run=True,
    )

    schedule = pd.read_csv(queue_dir / "schedule.csv")
    assert schedule["initial_assignment"].tolist() == [
        "cuda:0",
        "cuda:1",
        "next_available",
    ]
    assert not any(
        spec.batch_dir.joinpath("runs").exists() for spec in prepare_benchmark_specs(configs)
    )


def test_advisory_claim_prevents_duplicate_benchmark_ownership(tmp_path) -> None:
    spec = prepare_benchmark_specs([_write_benchmark(tmp_path, "claim")])[0]

    with (
        _benchmark_claim(spec, "cuda:0"),
        pytest.raises(BenchmarkClaimError, match="already claimed"),
        _benchmark_claim(spec, "cuda:1"),
    ):
        pass


def test_device_override_does_not_change_stable_run_identity(tmp_path) -> None:
    config_path = _write_benchmark(tmp_path, "identity")
    raw = _load_yaml(config_path)
    base_raw = _load_yaml(Path(raw["base_config"]))
    batch = raw["batch"]

    default_runs = expand_batch_runs(base_raw, batch, tmp_path / "default")
    cuda_runs = expand_batch_runs(
        base_raw,
        batch,
        tmp_path / "cuda",
        device_override="cuda",
    )

    assert [run.run_id for run in default_runs] == [run.run_id for run in cuda_runs]
    assert [run.config.experiment.seed for run in default_runs] == [
        run.config.experiment.seed for run in cuda_runs
    ]
    assert cuda_runs[0].config.experiment.device == "cuda"


def test_parallel_and_serial_neural_metrics_are_equivalent(tmp_path) -> None:
    configs = [
        _write_benchmark(tmp_path, "study_a", model="mlp"),
        _write_benchmark(tmp_path, "study_b", model="mlp"),
    ]
    serial_metrics = {}
    for config in configs:
        batch_dir = run_batch(config, device_override="cpu")
        serial_metrics[batch_dir.name] = _ranking_metrics(batch_dir)
        shutil.rmtree(batch_dir)

    run_benchmark_queue(configs, ["cpu", "cpu"])

    for config in configs:
        raw = _load_yaml(config)
        batch_dir = Path(raw["batch"]["output_dir"]) / raw["batch"]["name"]
        parallel_metrics = _ranking_metrics(batch_dir)
        np.testing.assert_allclose(
            parallel_metrics,
            serial_metrics[batch_dir.name],
            rtol=1e-7,
            atol=1e-8,
        )
        worker_state = _load_yaml(batch_dir / "worker_state.yaml")
        assert worker_state["status"] == "completed"
        assert worker_state["device"] == "cpu"
        metadata = _load_yaml(next((batch_dir / "runs").glob("*/metadata.yaml")))
        assert metadata["worker_device"] == "cpu"


def test_queue_combines_compatible_baseline_and_neural_reports(tmp_path) -> None:
    baseline = _write_benchmark(
        tmp_path,
        "baseline_slice",
        model="naive_persistence",
        seeds=[42],
    )
    neural = _write_benchmark(
        tmp_path,
        "neural_slice",
        model="mlp",
        seeds=[1, 42],
    )

    queue_dir = run_benchmark_queue([baseline, neural], ["cpu", "cpu"])

    status = _load_yaml(queue_dir / "combined_report_status.yaml")
    assert status["status"] == "completed"
    report_dir = queue_dir / "combined_report"
    results = pd.read_csv(report_dir / "run_results.csv")
    assert set(results["model"]) == {"naive_persistence", "mlp"}
    neural_results = results[results["model"] == "mlp"]
    assert neural_results["persistence_price_rmse"].notna().all()
    assert neural_results["price_rmse_ratio_to_persistence"].notna().all()
    assert neural_results["price_rmse_rank"].isin([1.0, 2.0]).all()

    aggregate = pd.read_csv(report_dir / "aggregate_metrics.csv")
    assert set(aggregate["model"]) == {"naive_persistence", "mlp"}
    mlp = aggregate[aggregate["model"] == "mlp"].iloc[0]
    assert mlp["seeds_completed"] == 2
    assert 0 <= mlp["seeds_beating_persistence"] <= 2
    assert (report_dir / "source_batches.csv").is_file()


def _write_benchmark(
    tmp_path: Path,
    name: str,
    *,
    batch_name: str | None = None,
    model: str = "naive_persistence",
    seeds: list[int] | None = None,
) -> Path:
    data_path = tmp_path / "prices.csv"
    if not data_path.exists():
        values = np.linspace(100.0, 125.0, 90)
        pd.DataFrame(
            {
                "Date": pd.date_range("2021-01-01", periods=len(values)),
                "Close": values + np.sin(np.arange(len(values))),
            }
        ).to_csv(data_path, index=False)
    base_path = tmp_path / "base.yaml"
    if not base_path.exists():
        base_path.write_text(
            yaml.safe_dump(
                {
                    "experiment": {
                        "name": "base",
                        "output_dir": str(tmp_path / "unused"),
                        "seed": 42,
                        "device": "cpu",
                    },
                    "dataset": {
                        "name": "synthetic",
                        "csv_path": str(data_path),
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
                        "num_workers": 0,
                        "models": [{"name": "mlp", "params": {"hidden_sizes": [4]}}],
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base_path),
                "batch": {
                    "name": batch_name or name,
                    "output_dir": str(tmp_path / "batches"),
                    "matched_origins": False,
                    "datasets": [
                        {
                            "name": "synthetic",
                            "csv_path": str(data_path),
                            "date_column": "Date",
                            "target_column": "Close",
                        }
                    ],
                    "models": [model],
                    "feature_sets": {
                        "close": {
                            "feature_columns": ["Close"],
                            "technical_indicators": None,
                        }
                    },
                    "seeds": seeds or [7],
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
    return config_path


def _ranking_metrics(batch_dir: Path) -> np.ndarray:
    frame = pd.read_csv(batch_dir / "run_results.csv")
    return frame.loc[0, ["price_rmse", "price_mae", "rmse", "mae"]].to_numpy(dtype=float)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)
