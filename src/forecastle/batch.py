from __future__ import annotations

import copy
import hashlib
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from forecastle.artifacts import write_dataframe
from forecastle.batch_analysis import validate_run_artifacts, write_batch_summaries
from forecastle.config import AppConfig, parse_config
from forecastle.experiment import run_experiment

BASELINE_NAMES = {"naive_persistence", "linear_regression"}


@dataclass(frozen=True)
class BatchRun:
    run_id: str
    market: str
    model: str
    feature_set: str
    seed: int
    config: AppConfig
    config_yaml: str
    config_sha256: str
    run_root: Path


def run_batch(config_path: Path, limit: int | None = None) -> Path:
    raw = _load_yaml(config_path)
    batch = _require_mapping(raw.get("batch"), "batch")
    batch_name = _slug(str(batch["name"]))
    batch_dir = Path(batch.get("output_dir", "outputs/batches")) / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    base_path = _resolve_path(config_path.parent, Path(raw["base_config"]))
    base_raw = _load_yaml(base_path)
    runs = expand_batch_runs(base_raw, batch, batch_dir)
    if limit is not None:
        if limit < 1:
            msg = "Batch limit must be positive."
            raise ValueError(msg)
        selected_runs = runs[:limit]
    else:
        selected_runs = runs

    (batch_dir / "batch_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_study_metadata(batch_dir, config_path, base_path, raw, len(runs))
    _write_planned_runs(batch_dir, runs)

    fail_fast = bool(batch.get("fail_fast", False))
    for batch_run in selected_runs:
        batch_run.run_root.mkdir(parents=True, exist_ok=True)
        (batch_run.run_root / "config.yaml").write_text(batch_run.config_yaml, encoding="utf-8")
        existing = _load_metadata(batch_run.run_root / "metadata.yaml")
        if _is_completed(batch_run, existing):
            print(f"SKIP {batch_run.run_id} (already completed)")
            continue

        started_at = _now()
        metadata = _base_run_metadata(batch_run)
        metadata.update({"status": "running", "started_at": started_at})
        _write_metadata(batch_run.run_root / "metadata.yaml", metadata)
        print(f"RUN  {batch_run.run_id}")
        start = time.perf_counter()
        try:
            result = run_experiment(batch_run.config)
            validate_run_artifacts(result.run_dir, batch_run.model)
        except KeyboardInterrupt:
            metadata.update(
                {
                    "status": "interrupted",
                    "completed_at": _now(),
                    "duration_seconds": time.perf_counter() - start,
                }
            )
            _write_metadata(batch_run.run_root / "metadata.yaml", metadata)
            write_batch_summaries(batch_dir)
            raise
        except Exception as error:
            metadata.update(
                {
                    "status": "failed",
                    "completed_at": _now(),
                    "duration_seconds": time.perf_counter() - start,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _write_metadata(batch_run.run_root / "metadata.yaml", metadata)
            print(f"FAIL {batch_run.run_id}: {type(error).__name__}: {error}")
            if fail_fast:
                write_batch_summaries(batch_dir)
                raise
        else:
            metadata.update(
                {
                    "status": "completed",
                    "completed_at": _now(),
                    "duration_seconds": time.perf_counter() - start,
                    "artifact_dir": str(result.run_dir),
                }
            )
            _write_metadata(batch_run.run_root / "metadata.yaml", metadata)

    write_batch_summaries(batch_dir)
    return batch_dir


def expand_batch_runs(
    base_raw: dict[str, Any],
    batch: dict[str, Any],
    batch_dir: Path,
) -> list[BatchRun]:
    datasets = _require_list(batch.get("datasets"), "batch.datasets")
    feature_sets = _require_mapping(batch.get("feature_sets"), "batch.feature_sets")
    models = [str(model) for model in _require_list(batch.get("models"), "batch.models")]
    seeds = [int(seed) for seed in _require_list(batch.get("seeds"), "batch.seeds")]
    _validate_unique(models, "batch.models")
    _validate_unique(seeds, "batch.seeds")

    training = _require_mapping(base_raw.get("training"), "base training")
    model_definitions = {
        str(item["name"]): item
        for item in _require_list(training.get("models"), "base training.models")
    }
    unknown = [model for model in models if model not in BASELINE_NAMES | model_definitions.keys()]
    if unknown:
        msg = f"Batch models are not defined by the base config: {', '.join(unknown)}."
        raise ValueError(msg)

    dataset_names = [str(_require_mapping(item, "batch dataset")["name"]) for item in datasets]
    _validate_unique(dataset_names, "batch dataset names")
    runs = []
    for dataset_item in datasets:
        dataset = _require_mapping(dataset_item, "batch dataset")
        market = str(dataset["name"])
        for model in models:
            for feature_name, feature_value in feature_sets.items():
                feature = _require_mapping(feature_value, f"feature set {feature_name}")
                for seed in seeds:
                    run_id = stable_run_id(market, model, str(feature_name), seed)
                    run_root = batch_dir / "runs" / run_id
                    run_raw = _make_run_raw(
                        base_raw,
                        batch,
                        dataset,
                        str(feature_name),
                        feature,
                        model,
                        model_definitions,
                        seed,
                        run_id,
                        run_root,
                    )
                    config = parse_config(run_raw)
                    config_yaml = yaml.safe_dump(run_raw, sort_keys=False)
                    runs.append(
                        BatchRun(
                            run_id=run_id,
                            market=market,
                            model=model,
                            feature_set=str(feature_name),
                            seed=seed,
                            config=config,
                            config_yaml=config_yaml,
                            config_sha256=_sha256_bytes(config_yaml.encode()),
                            run_root=run_root,
                        )
                    )
    run_ids = [run.run_id for run in runs]
    _validate_unique(run_ids, "stable batch run IDs")
    return runs


def stable_run_id(market: str, model: str, feature_set: str, seed: int) -> str:
    return "__".join((_slug(market), _slug(model), _slug(feature_set), f"seed{seed}"))


def _make_run_raw(
    base_raw: dict[str, Any],
    batch: dict[str, Any],
    dataset: dict[str, Any],
    feature_name: str,
    feature: dict[str, Any],
    model: str,
    model_definitions: dict[str, Any],
    seed: int,
    run_id: str,
    run_root: Path,
) -> dict[str, Any]:
    raw = copy.deepcopy(base_raw)
    raw.setdefault("experiment", {}).update(
        {"name": "artifacts", "output_dir": str(run_root), "seed": seed}
    )
    raw.setdefault("dataset", {}).update(dataset)
    raw["dataset"]["feature_columns"] = list(feature.get("feature_columns", ["Close"]))
    if "technical_indicators" in feature:
        indicators = feature["technical_indicators"]
        if indicators is None:
            raw["dataset"].pop("technical_indicators", None)
        else:
            raw["dataset"]["technical_indicators"] = copy.deepcopy(indicators)
    if "horizon" in batch:
        raw["dataset"]["horizon"] = int(batch["horizon"])
    raw.setdefault("forecasting", {}).update(
        _require_mapping(batch.get("forecasting", {}), "batch.forecasting")
    )
    raw.setdefault("evaluation", {}).update(
        _require_mapping(batch.get("evaluation", {}), "batch.evaluation")
    )
    raw["training"]["models"] = (
        [] if model in BASELINE_NAMES else [copy.deepcopy(model_definitions[model])]
    )
    raw["training"]["baselines"] = [model] if model in BASELINE_NAMES else []
    raw["batch_run"] = {
        "id": run_id,
        "market": str(dataset["name"]),
        "model": model,
        "feature_set": feature_name,
        "seed": seed,
    }
    return raw


def _base_run_metadata(batch_run: BatchRun) -> dict[str, Any]:
    dataset_path = batch_run.config.dataset.csv_path
    return {
        "run_id": batch_run.run_id,
        "market": batch_run.market,
        "model": batch_run.model,
        "feature_set": batch_run.feature_set,
        "seed": batch_run.seed,
        "config_sha256": batch_run.config_sha256,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "git_revision": _git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "forecastle_version": version("forecastle"),
        "torch_version": str(torch.__version__),
        "device_requested": batch_run.config.experiment.device,
    }


def _is_completed(batch_run: BatchRun, metadata: dict[str, Any] | None) -> bool:
    if metadata is None or metadata.get("status") != "completed":
        return False
    if metadata.get("config_sha256") != batch_run.config_sha256:
        return False
    artifact_dir = Path(str(metadata.get("artifact_dir", "")))
    try:
        validate_run_artifacts(artifact_dir, batch_run.model)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _write_study_metadata(
    batch_dir: Path,
    config_path: Path,
    base_path: Path,
    raw: dict[str, Any],
    run_count: int,
) -> None:
    payload = {
        "batch_config": str(config_path),
        "batch_config_sha256": _sha256_file(config_path),
        "base_config": str(base_path),
        "base_config_sha256": _sha256_file(base_path),
        "planned_runs": run_count,
        "git_revision": _git_revision(),
        "updated_at": _now(),
        "batch": raw.get("batch", {}),
    }
    _write_metadata(batch_dir / "study_metadata.yaml", payload)


def _write_planned_runs(batch_dir: Path, runs: list[BatchRun]) -> None:
    frame = pd.DataFrame(
        [
            {
                "run_id": run.run_id,
                "market": run.market,
                "model": run.model,
                "feature_set": run.feature_set,
                "seed": run.seed,
                "config_sha256": run.config_sha256,
            }
            for run in runs
        ]
    )
    write_dataframe(batch_dir / "planned_runs.csv", frame)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        msg = f"Config file {path} must contain a YAML mapping."
        raise ValueError(msg)
    return raw


def _load_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_yaml(path)


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{name} must be a YAML mapping."
        raise ValueError(msg)
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty YAML list."
        raise ValueError(msg)
    return value


def _validate_unique(values: list[Any], name: str) -> None:
    if len(values) != len(set(values)):
        msg = f"{name} must not contain duplicates."
        raise ValueError(msg)


def _resolve_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        msg = f"Cannot form a stable identifier from {value!r}."
        raise ValueError(msg)
    return slug


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()
