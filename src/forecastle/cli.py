from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run financial forecasting experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from a YAML config.")
    run_parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")

    batch_parser = subparsers.add_parser(
        "batch", help="Run or resume a batch of experiments from YAML."
    )
    batch_parser.add_argument(
        "--config",
        required=True,
        action="extend",
        nargs="+",
        type=Path,
        help="One or more batch YAML configs. Repeat the option or provide multiple paths.",
    )
    batch_parser.add_argument(
        "--devices",
        type=_device_list,
        default=None,
        help="Comma-separated worker devices, for example cuda:0,cuda:1.",
    )
    batch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of stable batch runs to consider.",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the batch matrix or worker schedule without training models.",
    )
    batch_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry unchanged failed runs instead of preserving their recorded failure.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two completed matched-origin batches.",
    )
    compare_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to comparison YAML config.",
    )

    sweep_parser = subparsers.add_parser("sweep", help="Run an experiment sweep from YAML.")
    sweep_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to sweep YAML config.",
    )
    sweep_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of sweep variants to run.",
    )

    tune_parser = subparsers.add_parser("tune", help="Tune one neural model with Optuna.")
    tune_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to experiment YAML config containing a tuning section.",
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download OHLCV data from Yahoo Finance via yfinance.",
    )
    download_source = download_parser.add_mutually_exclusive_group(required=True)
    download_source.add_argument("--symbol", help="Yahoo Finance symbol, for example ^GSPC.")
    download_source.add_argument(
        "--dataset",
        help="Dataset key from a downloads YAML file, for example sp500.",
    )
    download_parser.add_argument(
        "--downloads-config",
        type=Path,
        default=Path("configs/downloads.yaml"),
        help="YAML file containing dataset download presets.",
    )
    download_parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Required with --symbol; optional with --dataset.",
    )
    download_parser.add_argument("--start", default="2000-01-01", help="Start date, YYYY-MM-DD.")
    download_parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD.")
    download_parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval, for example 1d.",
    )
    download_parser.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Use yfinance adjusted OHLC prices.",
    )

    args = parser.parse_args()
    if args.command == "run":
        from forecastle.config import load_config
        from forecastle.experiment import run_experiment

        config = load_config(args.config)
        result = run_experiment(config)
        print(f"Wrote experiment artifacts to {result.run_dir}")
    elif args.command == "batch":
        if args.devices is not None or len(args.config) > 1:
            if args.devices is None:
                parser.error("--devices is required when multiple batch configs are provided.")
            from forecastle.benchmark_queue import run_benchmark_queue

            queue_dir = run_benchmark_queue(
                args.config,
                args.devices,
                limit=args.limit,
                dry_run=args.dry_run,
                retry_failed=args.retry_failed,
            )
            print(f"Wrote benchmark queue artifacts to {queue_dir}")
        else:
            from forecastle.batch import run_batch

            batch_dir = run_batch(
                args.config[0],
                limit=args.limit,
                dry_run=args.dry_run,
                retry_failed=args.retry_failed,
            )
            print(f"Wrote batch artifacts to {batch_dir}")
    elif args.command == "compare":
        from forecastle.comparison import run_comparison

        comparison_dir = run_comparison(args.config)
        print(f"Wrote comparison artifacts to {comparison_dir}")
    elif args.command == "sweep":
        from forecastle.sweep import run_sweep

        sweep_dir = run_sweep(args.config, limit=args.limit)
        print(f"Wrote sweep artifacts to {sweep_dir}")
    elif args.command == "tune":
        from forecastle.config import load_config
        from forecastle.tuning import run_tuning

        config = load_config(args.config)
        tune_dir = run_tuning(config)
        print(f"Wrote tuning artifacts to {tune_dir}")
    elif args.command == "download":
        from forecastle.data.downloader import (
            DownloadRequest,
            download_prices,
            download_prices_from_config,
        )

        if args.symbol is not None:
            if args.output is None:
                parser.error("--output is required when using --symbol.")
            output_path = download_prices(
                DownloadRequest(
                    symbol=args.symbol,
                    output_path=args.output,
                    start=args.start,
                    end=args.end,
                    interval=args.interval,
                    auto_adjust=args.auto_adjust,
                )
            )
        else:
            output_path = download_prices_from_config(
                args.downloads_config,
                args.dataset,
                start=args.start,
                end=args.end,
                interval=args.interval,
                auto_adjust=args.auto_adjust,
                output_path=args.output,
            )
        print(f"Wrote downloaded prices to {output_path}")


def _device_list(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("At least one device is required.")
    return devices
