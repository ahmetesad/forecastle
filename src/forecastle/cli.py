from __future__ import annotations

import argparse
from pathlib import Path

from forecastle.config import load_config
from forecastle.data.downloader import DownloadRequest, download_prices, download_prices_from_config
from forecastle.experiment import run_experiment
from forecastle.sweep import run_sweep
from forecastle.tuning import run_tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Run financial forecasting experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from a YAML config.")
    run_parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")

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
        config = load_config(args.config)
        result = run_experiment(config)
        print(f"Wrote experiment artifacts to {result.run_dir}")
    elif args.command == "sweep":
        sweep_dir = run_sweep(args.config, limit=args.limit)
        print(f"Wrote sweep artifacts to {sweep_dir}")
    elif args.command == "tune":
        config = load_config(args.config)
        tune_dir = run_tuning(config)
        print(f"Wrote tuning artifacts to {tune_dir}")
    elif args.command == "download":
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
