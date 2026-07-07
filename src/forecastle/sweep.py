from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from forecastle.artifacts import dataframe_to_markdown
from forecastle.config import DatasetConfig, load_config
from forecastle.experiment import run_experiment


def run_sweep(config_path: Path, limit: int | None = None) -> Path:
    raw = load_sweep_yaml(config_path)
    base_config_path = resolve_path(config_path.parent, Path(raw["base_config"]))
    base_config = load_config(base_config_path)
    sweep = raw.get("sweep", {})
    sweep_name = str(sweep.get("name", "sweep"))
    sweep_dir = make_sweep_dir(Path(sweep.get("output_dir", "outputs/sweeps")), sweep_name)

    rows: list[dict[str, Any]] = []
    variants_ran = 0
    for dataset_name, dataset_config in iter_dataset_configs(base_config.dataset, sweep):
        for feature_set_name, feature_columns in iter_feature_sets(dataset_config, sweep):
            for sequence_length in sweep.get("sequence_lengths", [dataset_config.sequence_length]):
                for horizon in sweep.get("horizons", [dataset_config.horizon]):
                    if limit is not None and variants_ran >= limit:
                        write_sweep_results(sweep_dir, rows)
                        return sweep_dir
                    variant_name = (
                        f"{dataset_name}_{feature_set_name}_"
                        f"lookback{sequence_length}_horizon{horizon}"
                    )
                    variant_config = replace(
                        base_config,
                        experiment=replace(base_config.experiment, name=variant_name),
                        dataset=replace(
                            dataset_config,
                            feature_columns=list(feature_columns),
                            sequence_length=int(sequence_length),
                            horizon=int(horizon),
                        ),
                    )
                    result = run_experiment(variant_config)
                    variants_ran += 1
                    for row in result.comparison_rows:
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "feature_set": feature_set_name,
                                "sequence_length": int(sequence_length),
                                "horizon": int(horizon),
                                "run_dir": str(result.run_dir),
                                **row,
                            }
                        )

    write_sweep_results(sweep_dir, rows)
    return sweep_dir


def load_sweep_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        msg = f"Sweep config file {config_path} must contain a YAML mapping."
        raise ValueError(msg)
    return raw


def iter_dataset_configs(
    base_dataset: DatasetConfig,
    sweep: dict[str, Any],
) -> list[tuple[str, DatasetConfig]]:
    datasets = sweep.get("datasets")
    if not datasets:
        return [(base_dataset.name, base_dataset)]
    parsed = []
    for item in datasets:
        dataset = replace(
            base_dataset,
            name=str(item.get("name", base_dataset.name)),
            csv_path=Path(item.get("csv_path", base_dataset.csv_path)),
            date_column=str(item.get("date_column", base_dataset.date_column)),
            target_column=str(item.get("target_column", base_dataset.target_column)),
        )
        parsed.append((dataset.name, dataset))
    return parsed


def iter_feature_sets(
    dataset_config: DatasetConfig,
    sweep: dict[str, Any],
) -> list[tuple[str, list[str]]]:
    feature_sets = sweep.get("feature_sets")
    if not feature_sets:
        return [("configured", dataset_config.feature_columns or [])]
    return [(str(name), list(columns)) for name, columns in feature_sets.items()]


def make_sweep_dir(output_dir: Path, sweep_name: str) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    sweep_dir = output_dir / sweep_name / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=False)
    return sweep_dir


def write_sweep_results(sweep_dir: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    if rows:
        frame = frame.sort_values(["dataset", "feature_set", "horizon", "sequence_length", "rmse"])
    frame.to_csv(sweep_dir / "sweep_results.csv", index=False)
    (sweep_dir / "sweep_results.md").write_text(dataframe_to_markdown(frame), encoding="utf-8")


def resolve_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path
