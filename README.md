# forecastle

`forecastle` is a modular PyTorch project for comparing neural network architectures on
financial time series forecasting tasks.

The project is built around independent dataset, model, training, and evaluation modules.
Experiments are configured with YAML, seeded for reproducibility, and write checkpoints,
predictions, plots, metrics, and comparison tables under `outputs/`. The default WIG20
example predicts log returns and includes a naive persistence baseline.

## Quick start

```bash
uv sync --dev
uv run forecastle download --dataset sp500
uv run forecastle run --config configs/example.yaml
```

Use `configs/example.yaml` with your own CSV paths. The expected CSV format is flexible:
set `date_column`, `target_column`, and optional `feature_columns` in YAML.
Set `target_transform` to `price`, `return`, or `log_return`.

## Downloading data

The project includes a Yahoo Finance downloader powered by `yfinance`.

Download one of the presets in `configs/downloads.yaml`:

```bash
uv run forecastle download --dataset sp500
uv run forecastle download --dataset wig20
uv run forecastle download --dataset bist100
```

Download a custom symbol:

```bash
uv run forecastle download --symbol ^GSPC --output data/raw/sp500.csv
```

Choose a date range or interval:

```bash
uv run forecastle download --dataset sp500 --start 2010-01-01 --end 2024-12-31
uv run forecastle download --symbol AAPL --output data/raw/aapl.csv --interval 1wk
```

The preset symbols are:

- WIG20: `WIG20.WA`
- BIST100: `XU100.IS`
- S&P 500: `^GSPC`

Downloaded CSVs contain the columns expected by the example dataset configs:
`Date`, `Open`, `High`, `Low`, `Close`, optional `Adj Close`, and `Volume`.

## Project layout

```text
configs/              Example experiment and dataset configs
src/forecastle/       Package source
tests/                Focused unit tests
outputs/              Experiment artifacts, ignored by git
data/raw/             Local financial CSV files, ignored by git
```

## Adding a dataset

1. Download or put a CSV in `data/raw/`, for example `data/raw/sp500.csv`.
2. Add a dataset YAML file under `configs/datasets/`.
3. Reference that dataset from an experiment YAML.

The built-in CSV loader handles sorting by date, feature selection, scaling, rolling windows,
and train/validation/test splits.

## Supported initial models

- MLP
- RNN
- LSTM
- GRU
- 1D CNN
- DNFS baseline

## Generated artifacts

Each run writes:

- `checkpoints/<model>.pt`
- `predictions/<model>_predictions.csv`
- `plots/<model>_predictions.png`
- `metrics/<model>_metrics.yaml`
- `comparison.csv`
- `comparison.md`

The comparison includes two non-neural baselines:

- `naive_persistence`: for price targets, tomorrow equals today; for return targets, next
  return is zero.
- `linear_regression`: flattened lookback windows fit with ordinary least squares on the
  training split only.

## DNFS baseline

The `dnfs` model is a compact neuro-fuzzy regression baseline inspired by ANFIS-style
Takagi-Sugeno systems.

Example YAML:

```yaml
- name: dnfs
  params:
    num_rules: 16
    dropout: 0.1
```

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

## Hyperparameter tuning

Tune an LSTM with Optuna:

```bash
uv run forecastle tune --config configs/tuning/lstm_wig20.yaml
```

The tuning command reuses the normal dataset, model, and training pipeline. It preserves the
chronological train/validation/test split, fits scalers on the training split only, and optimizes
the selected model on validation data. The example config searches over hidden size, number of
layers, dropout, learning rate, weight decay, batch size, and lookback length.

Configure tuning in YAML:

```yaml
tuning:
  enabled: true
  model: lstm
  trials: 50
  metric: price_rmse
  seed: 42
  study_name: wig20_lstm_optuna
  storage: sqlite:///outputs/studies/wig20_lstm_optuna.db
  sequence_lengths: [7, 14, 30, 60, 120]
  batch_sizes: [16, 32, 64, 128]
```

Set `metric` to `rmse` to optimize validation RMSE in target space, or `price_rmse` to optimize
RMSE after reconstructing prices from return/log-return predictions.
Use a fixed `study_name` and `storage` path when you want interrupted tuning runs to resume
instead of starting a new study.

Each tuning run writes artifacts under `outputs/<experiment>_tuning/<timestamp>/`:

- `study.db`: SQLite Optuna study database, unless a custom `tuning.storage` is configured.
- `best_params.yaml`: best Optuna parameter set.
- `best_summary.yaml`: best validation score, validation persistence baseline, timing, and trial
  metadata.
- `optimization_history.csv` and `optimization_history.md`: all trial results.
- `top_trials.csv` and `top_trials.md`: the best complete trials with key hyperparameters,
  validation metrics, and matching persistence scores.
- `parameter_importance.yaml`: Optuna parameter importance values when available.
- `optimization_history.png` and `parameter_importance.png`: diagnostic plots.
- `tuned_config.yaml`: ready-to-run experiment config containing the best parameters.

Run the tuned configuration as a normal experiment:

```bash
uv run forecastle run --config outputs/wig20_lstm_tuning/<timestamp>/tuned_config.yaml
```

## Sweeps

Run the first-pass WIG20 experiment grid:

```bash
uv run forecastle sweep --config configs/sweeps/wig20_features_lookbacks_horizons.yaml
```

This compares Close-only versus OHLCV inputs, lookbacks of `7`, `14`, `30`, `60`, and `120`
days, and horizons of `1`, `5`, `10`, and `20` days. Technical indicators are intentionally
left out of this first pass.

For a quick smoke check, cap the number of variants:

```bash
uv run forecastle sweep --config configs/sweeps/wig20_features_lookbacks_horizons.yaml --limit 1
```

After downloading `sp500` and `bist100`, run the cross-market grid:

```bash
uv run forecastle download --dataset sp500
uv run forecastle download --dataset bist100
uv run forecastle sweep --config configs/sweeps/markets_lookbacks_horizons.yaml
```
