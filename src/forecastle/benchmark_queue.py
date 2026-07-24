from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import multiprocessing as mp
import os
import queue
import re
import socket
import tempfile
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


class BenchmarkQueueError(RuntimeError):
    """Raised when one or more queued benchmarks cannot be completed."""


class BenchmarkClaimError(RuntimeError):
    """Raised when another process already owns a benchmark directory."""


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    config_path: Path
    config_sha256: str
    batch_name: str
    batch_dir: Path


@dataclass(frozen=True)
class WorkerOptions:
    limit: int | None
    retry_failed: bool
    queue_id: str


def run_benchmark_queue(
    config_paths: list[Path],
    devices: list[str],
    *,
    limit: int | None = None,
    dry_run: bool = False,
    retry_failed: bool = False,
) -> Path:
    """Run independent batch studies through persistent, device-pinned workers."""
    specs = prepare_benchmark_specs(config_paths)
    normalized_devices = normalize_devices(devices)
    queue_dir = _queue_dir(specs)
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_id = queue_dir.name
    schedule_rows = _schedule_rows(specs, normalized_devices)
    _write_csv(queue_dir / "schedule.csv", schedule_rows)
    _write_yaml(
        queue_dir / "queue_metadata.yaml",
        {
            "queue_id": queue_id,
            "created_at": _now(),
            "devices": normalized_devices,
            "benchmark_count": len(specs),
            "configs": [str(spec.config_path) for spec in specs],
            "dynamic_dispatch": True,
        },
    )
    _print_schedule(schedule_rows)
    if dry_run:
        return queue_dir

    context = mp.get_context("spawn")
    task_queue = context.JoinableQueue()
    result_queue = context.Queue()
    options = WorkerOptions(
        limit=limit,
        retry_failed=retry_failed,
        queue_id=queue_id,
    )
    workers = [
        context.Process(
            target=_worker_main,
            args=(task_queue, result_queue, device, options),
            name=f"forecastle-{_device_slug(device)}",
        )
        for device in normalized_devices
    ]
    for worker in workers:
        worker.start()
    for spec in specs:
        task_queue.put(spec)
    for _ in workers:
        task_queue.put(None)

    states = {
        spec.benchmark_id: {
            "benchmark_id": spec.benchmark_id,
            "config_path": str(spec.config_path),
            "batch_dir": str(spec.batch_dir),
            "status": "queued",
            "device": "",
            "started_at": "",
            "completed_at": "",
            "error": "",
            "log_path": "",
        }
        for spec in specs
    }
    _write_state(queue_dir, states)
    terminal_count = 0
    try:
        while terminal_count < len(specs):
            try:
                event = result_queue.get(timeout=1.0)
            except queue.Empty:
                if any(worker.is_alive() for worker in workers):
                    continue
                for state in states.values():
                    if state["status"] in {"queued", "running"}:
                        state.update(
                            {
                                "status": "failed",
                                "completed_at": _now(),
                                "error": "Worker exited without reporting a terminal result.",
                            }
                        )
                        terminal_count += 1
                _write_state(queue_dir, states)
                break
            benchmark_id = str(event["benchmark_id"])
            states[benchmark_id].update(event)
            if event["status"] == "running":
                print(f"START {benchmark_id} on {event['device']}")
            else:
                terminal_count += 1
                print(f"{event['status'].upper():<8} {benchmark_id} on {event['device']}")
            _write_state(queue_dir, states)
    except KeyboardInterrupt:
        for worker in workers:
            worker.terminate()
        raise
    finally:
        for worker in workers:
            worker.join()
        task_queue.close()
        result_queue.close()

    failed = [state for state in states.values() if state["status"] != "completed"]
    if failed:
        names = ", ".join(str(state["benchmark_id"]) for state in failed)
        msg = f"{len(failed)} queued benchmark(s) did not complete: {names}. See {queue_dir}."
        raise BenchmarkQueueError(msg)
    _write_combined_queue_report(queue_dir, specs)
    return queue_dir


def prepare_benchmark_specs(config_paths: list[Path]) -> list[BenchmarkSpec]:
    if not config_paths:
        msg = "At least one batch configuration is required."
        raise ValueError(msg)
    specs = []
    seen_configs: set[Path] = set()
    seen_batch_dirs: dict[Path, Path] = {}
    for config_value in config_paths:
        config_path = config_value.resolve()
        if config_path in seen_configs:
            msg = f"Batch configuration was provided more than once: {config_path}."
            raise ValueError(msg)
        seen_configs.add(config_path)
        raw = _load_yaml(config_path)
        batch = _require_mapping(raw.get("batch"), "batch")
        batch_name = _slug(str(batch["name"]))
        batch_dir = (Path(batch.get("output_dir", "outputs/batches")) / batch_name).resolve()
        previous = seen_batch_dirs.get(batch_dir)
        if previous is not None:
            msg = (
                f"Batch configurations must use separate output directories: "
                f"{previous} and {config_path} both resolve to {batch_dir}."
            )
            raise ValueError(msg)
        seen_batch_dirs[batch_dir] = config_path
        config_sha256 = _sha256_file(config_path)
        specs.append(
            BenchmarkSpec(
                benchmark_id=f"{batch_name}__{config_sha256[:10]}",
                config_path=config_path,
                config_sha256=config_sha256,
                batch_name=batch_name,
                batch_dir=batch_dir,
            )
        )
    return specs


def normalize_devices(devices: list[str]) -> list[str]:
    if not devices:
        msg = "At least one worker device is required."
        raise ValueError(msg)
    normalized = []
    for value in devices:
        device = value.strip().lower()
        if device.isdigit():
            device = f"cuda:{device}"
        if device != "cpu" and re.fullmatch(r"cuda:\d+", device) is None:
            msg = f"Worker device must be a physical CUDA device such as cuda:0, got {value!r}."
            raise ValueError(msg)
        normalized.append(device)
    cuda_devices = [device for device in normalized if device != "cpu"]
    if len(cuda_devices) != len(set(cuda_devices)):
        msg = "Worker devices must not contain duplicates."
        raise ValueError(msg)
    return normalized


def _worker_main(
    task_queue: Any,
    result_queue: Any,
    device: str,
    options: WorkerOptions,
) -> None:
    device_override = _configure_worker_device(device)
    from forecastle.batch import run_batch

    while True:
        spec = task_queue.get()
        if spec is None:
            task_queue.task_done()
            break
        result_queue.put(
            {
                "benchmark_id": spec.benchmark_id,
                "status": "running",
                "device": device,
                "started_at": _now(),
            }
        )
        log_dir = spec.batch_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{options.queue_id}__{_device_slug(device)}.log"
        temp_dir = spec.batch_dir / "tmp" / options.queue_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        status_path = spec.batch_dir / "worker_state.yaml"
        try:
            with (
                _benchmark_claim(spec, device),
                log_path.open("a", encoding="utf-8", buffering=1) as log_file,
                contextlib.redirect_stdout(log_file),
                contextlib.redirect_stderr(log_file),
                _benchmark_environment(temp_dir),
            ):
                _write_yaml(
                    status_path,
                    {
                        "benchmark_id": spec.benchmark_id,
                        "status": "running",
                        "device": device,
                        "pid": os.getpid(),
                        "started_at": _now(),
                        "config_path": str(spec.config_path),
                        "config_sha256": spec.config_sha256,
                        "log_path": str(log_path),
                        "temp_dir": str(temp_dir),
                    },
                )
                run_batch(
                    spec.config_path,
                    limit=options.limit,
                    retry_failed=options.retry_failed,
                    device_override=device_override,
                )
        except Exception as error:
            error_text = "".join(traceback.format_exception(error))
            _append_error(log_path, error_text)
            event = {
                "benchmark_id": spec.benchmark_id,
                "status": "failed",
                "device": device,
                "completed_at": _now(),
                "error": f"{type(error).__name__}: {error}",
                "log_path": str(log_path),
            }
            _write_yaml(status_path, event)
            result_queue.put(event)
        else:
            event = {
                "benchmark_id": spec.benchmark_id,
                "status": "completed",
                "device": device,
                "completed_at": _now(),
                "error": "",
                "log_path": str(log_path),
            }
            _write_yaml(status_path, event)
            result_queue.put(event)
        finally:
            task_queue.task_done()


def _configure_worker_device(device: str) -> str:
    os.environ["FORECASTLE_WORKER_DEVICE"] = device
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORECASTLE_PHYSICAL_GPU_ID"] = ""
        return "cpu"
    physical_gpu = device.split(":", maxsplit=1)[1]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    os.environ["FORECASTLE_PHYSICAL_GPU_ID"] = physical_gpu
    return "cuda"


@contextlib.contextmanager
def _benchmark_environment(temp_dir: Path):
    names = ("TMPDIR", "TMP", "TEMP")
    previous = {name: os.environ.get(name) for name in names}
    previous_tempdir = tempfile.tempdir
    try:
        for name in names:
            os.environ[name] = str(temp_dir)
        tempfile.tempdir = str(temp_dir)
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def _benchmark_claim(spec: BenchmarkSpec, device: str):
    claim_path = spec.batch_dir / ".benchmark.claim"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    with claim_path.open("a+", encoding="utf-8") as claim_file:
        try:
            fcntl.flock(claim_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            msg = f"Benchmark output is already claimed: {spec.batch_dir}."
            raise BenchmarkClaimError(msg) from error
        claim_file.seek(0)
        claim_file.truncate()
        yaml.safe_dump(
            {
                "benchmark_id": spec.benchmark_id,
                "device": device,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "claimed_at": _now(),
            },
            claim_file,
            sort_keys=False,
        )
        claim_file.flush()
        try:
            yield
        finally:
            fcntl.flock(claim_file.fileno(), fcntl.LOCK_UN)


def _queue_dir(specs: list[BenchmarkSpec]) -> Path:
    identity = "\n".join(
        f"{spec.config_path}:{spec.config_sha256}"
        for spec in sorted(specs, key=lambda item: item.benchmark_id)
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    output_parents = {spec.batch_dir.parent for spec in specs}
    queue_root = (
        output_parents.pop() / "_queues"
        if len(output_parents) == 1
        else Path("outputs/benchmark_queues")
    )
    return queue_root / f"queue_{digest}"


def _schedule_rows(
    specs: list[BenchmarkSpec],
    devices: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for position, spec in enumerate(specs):
        initial_device = devices[position] if position < len(devices) else "next_available"
        rows.append(
            {
                "queue_position": position + 1,
                "benchmark_id": spec.benchmark_id,
                "config_path": str(spec.config_path),
                "batch_dir": str(spec.batch_dir),
                "initial_assignment": initial_device,
            }
        )
    return rows


def _print_schedule(rows: list[dict[str, Any]]) -> None:
    print("Dynamic benchmark schedule:")
    for row in rows:
        print(
            f"  {row['queue_position']:>2}. {row['initial_assignment']:<14} {row['benchmark_id']}"
        )


def _write_state(queue_dir: Path, states: dict[str, dict[str, Any]]) -> None:
    rows = [states[key] for key in sorted(states)]
    _write_csv(queue_dir / "queue_state.csv", rows)
    _write_yaml(
        queue_dir / "queue_state.yaml",
        {"updated_at": _now(), "benchmarks": rows},
    )


def _write_combined_queue_report(queue_dir: Path, specs: list[BenchmarkSpec]) -> None:
    status_path = queue_dir / "combined_report_status.yaml"
    if len(specs) < 2:
        _write_yaml(
            status_path,
            {"status": "not_applicable", "reason": "Only one benchmark was queued."},
        )
        return
    signatures = {_reporting_signature(spec) for spec in specs}
    if len(signatures) != 1:
        _write_yaml(
            status_path,
            {
                "status": "not_applicable",
                "reason": "Queued benchmarks use different evaluation protocols.",
            },
        )
        return
    duplicate_run_ids = _duplicate_planned_run_ids(specs)
    if duplicate_run_ids:
        _write_yaml(
            status_path,
            {
                "status": "not_applicable",
                "reason": "Queued benchmarks contain duplicate physical run identities.",
                "duplicate_run_ids": duplicate_run_ids[:20],
            },
        )
        return

    from forecastle.batch_analysis import write_combined_batch_summaries

    report_dir = queue_dir / "combined_report"
    try:
        write_combined_batch_summaries(report_dir, [spec.batch_dir for spec in specs])
    except Exception as error:
        _write_yaml(
            status_path,
            {
                "status": "failed",
                "report_dir": str(report_dir),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        msg = f"Queued benchmarks completed, but combined reporting failed: {error}"
        raise BenchmarkQueueError(msg) from error
    _write_yaml(
        status_path,
        {
            "status": "completed",
            "report_dir": str(report_dir),
            "source_batches": [str(spec.batch_dir) for spec in specs],
        },
    )


def _reporting_signature(spec: BenchmarkSpec) -> str:
    raw = _load_yaml(spec.config_path)
    batch = _require_mapping(raw.get("batch"), "batch")
    base_value = Path(str(raw["base_config"]))
    base_path = (
        base_value.resolve()
        if base_value.is_absolute() or base_value.exists()
        else (spec.config_path.parent / base_value).resolve()
    )
    payload = {
        "base_config_sha256": _sha256_file(base_path),
        "matched_origins": batch.get("matched_origins", False),
        "datasets": batch.get("datasets"),
        "feature_sets": batch.get("feature_sets"),
        "horizon": batch.get("horizon"),
        "forecasting": batch.get("forecasting"),
        "evaluation": batch.get("evaluation"),
        "origin_schedule_sources": batch.get("origin_schedule_sources"),
    }
    serialized = yaml.safe_dump(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _duplicate_planned_run_ids(specs: list[BenchmarkSpec]) -> list[str]:
    run_ids = []
    for spec in specs:
        path = spec.batch_dir / "planned_runs.csv"
        if not path.is_file():
            return []
        with path.open(encoding="utf-8", newline="") as file:
            run_ids.extend(str(row["run_id"]) for row in csv.DictReader(file))
    counts: dict[str, int] = {}
    for run_id in run_ids:
        counts[run_id] = counts.get(run_id, 0) + 1
    return sorted(run_id for run_id, count in counts.items() if count > 1)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _append_error(path: Path, error_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(error_text)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        msg = f"Config file {path} must contain a YAML mapping."
        raise ValueError(msg)
    return raw


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{name} must be a YAML mapping."
        raise ValueError(msg)
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        msg = f"Cannot form a stable identifier from {value!r}."
        raise ValueError(msg)
    return slug


def _device_slug(device: str) -> str:
    return device.replace(":", "_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()
