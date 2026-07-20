# forecastle

`forecastle` is a modular PyTorch project for comparing neural network architectures on
financial time series forecasting tasks.

The project is built around independent dataset, model, training, and evaluation modules.
Experiments are configured with YAML, seeded for reproducibility, and write checkpoints,
predictions, plots, metrics, and comparison tables under `outputs/`. The default WIG20
example predicts log returns and includes a naive persistence baseline.

See [`STATUS.md`](STATUS.md) for implemented capabilities, curated findings, known limitations,
and the next research tasks.

## Quick start

```bash
uv sync --frozen --dev
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

## Included research datasets

The repository includes the exact CSV snapshots used by the existing experiment configs. Market
data would normally remain outside version control and be downloaded locally. These snapshots are
committed specifically because Forecastle is a research project being shared with collaborators;
including them makes it easier for everyone to run the same experiments against identical inputs.

| Dataset | File | Source | Coverage | Rows | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| WIG20 | `data/raw/wig20.csv` | Stooq daily CSV | 2006-07-07 to 2026-07-06 | 4,999 | `2068449fa9044aae9b8dd670428feb5e3e2fb7c35207f8d46027f83ee2cf5ce4` |
| S&P 500 | `data/raw/sp500.csv` | Yahoo Finance (`^GSPC`) | 2000-01-03 to 2026-07-08 | 6,667 | `973bb1b4c920de56183ef6404fdce8fa6bb7a5141263797a39139bcbfcd290c2` |
| BIST100 | `data/raw/bist100.csv` | Yahoo Finance (`XU100.IS`) | 2000-01-04 to 2026-07-08 | 6,631 | `73bebf639bf279a1d795e70709709369f3cb6951b64b976067c12a77b76d2aa9` |

Running a downloader command can
replace a snapshot with newer or provider-adjusted data and therefore change experiment results.
Use the committed files when reproducing the published tables.

## Project layout

```text
configs/              Example experiment and dataset configs
src/forecastle/       Package source
tests/                Focused unit tests
outputs/              Experiment artifacts, ignored by git
data/raw/             Committed research snapshots
```

## Adding a dataset

1. Download or put a CSV in `data/raw/`; the three built-in datasets are already included.
2. Add a dataset YAML file under `configs/datasets/`.
3. Reference that dataset from an experiment YAML.

The built-in CSV loader handles sorting by date, feature selection, scaling, rolling windows,
and train/validation/test splits.

## Supported models

- MLP
- RNN
- LSTM
- GRU
- 1D CNN
- Temporal DNFS (legacy flattened mode remains available)
- LSTM-GRU (`LSTM -> GRU -> linear head`)
- CNN-LSTM (`Conv1d -> LSTM -> linear head`)

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

## Evaluation strategies

The default remains the original chronological train/validation/test holdout. Enable
walk-forward evaluation explicitly:

```yaml
evaluation:
  strategy: walk_forward
  window: expanding  # or rolling
  step_size: null     # defaults to dataset.horizon
  validation_size: null
  train_window_size: null
  max_folds: null
```

Each fold ends at a forecast origin representing the last observed date. Forecastle fits fresh
training-only scalers and fresh neural models, uses the latest historical block for validation,
forecasts the next block, and advances. Expanding windows retain all older training data; rolling
windows retain a fixed training history. Use `max_folds` for development and smoke runs.

Run the included examples:

```bash
uv run forecastle run --config configs/evaluation/wig20_walk_forward_recursive.yaml
uv run forecastle run --config configs/evaluation/wig20_rolling_recursive.yaml
uv run forecastle run --config configs/evaluation/wig20_walk_forward_recursive_smoke.yaml
```

## Direct and recursive forecasting

Direct forecasting is the backward-compatible default and predicts only the endpoint at `t+h`.
Recursive forecasting trains a one-step model and feeds each prediction into the next input:

```yaml
forecasting:
  strategy: recursive
```

For simple returns, prices are reconstructed as `P(t+1) = P(t) * (1 + return)`. For log returns,
Forecastle uses `P(t+1) = P(t) * exp(log_return)`. Persistence predicts the last Close for price
targets and zero for return targets, then follows the same reconstruction path.

True future OHLCV values do not exist during recursive inference. Recursive configs therefore
accept only `feature_columns: [Close]` plus generated Close-derived indicators. Forecastle rejects
recursive OHLCV or exogenous-feature configs instead of silently leaking future information.

## Technical indicators

Configure causal Close-derived features under the dataset:

```yaml
technical_indicators:
  sma_periods: [5, 10, 15, 20]
  rsi_period: 14
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
```

This adds trailing SMAs, Wilder-style RSI, and the MACD line, signal, and histogram. Indicator
warm-up rows are removed before split boundaries are calculated. Every fold still fits feature and
target scalers on its training portion only.

## Walk-forward artifacts

Walk-forward runs preserve the normal comparison and per-model outputs and additionally write:

- `folds.csv` and `folds.md`
- `fold_metrics.csv` and `fold_metrics.md`
- `horizon_metrics.csv` and `horizon_metrics.md`
- `fit_summaries.csv` and `fit_summaries.md`
- `plots/horizon_rmse.png`
- fold checkpoints under `checkpoints/<model>/`
- DNFS rule diagnostics under `rule_analysis/rule_activations.csv`
- DNFS regime plot under `plots/dnfs_rule_activations.png`

Prediction files are long-form and uniquely keyed by model, fold, forecast origin, target date, and
horizon step. Direct forecasts contain only the configured endpoint; recursive forecasts contain
every generated step.

## Temporal DNFS

The `dnfs` model is a temporal deep neuro-fuzzy system. A GRU, LSTM, causal CNN1D, or legacy
flattening encoder maps each input window to a normalized latent state `z`. Rule `r` has a learned
Gaussian center `c[r]` and bounded positive width `sigma[r]`:

```text
log_strength[r] = -0.5 * reduce(((z - c[r]) / sigma[r])^2)
alpha = softmax(log_strength / rule_temperature)
prediction = sum_r alpha[r] * consequent[r](z)
```

`strength_reduction: mean` tempers the score by latent dimension; `sum` is the literal Gaussian
membership product evaluated in log space. Ordinary softmax gating uses every rule. Optional top-k
gating retains and renormalizes only the strongest rules.

Consequents may be `zero_order` constants, standard first-order affine Takagi-Sugeno functions, or
small rule-specific `mlp` networks. The MLP option is a generalized nonlinear TSK-style consequent;
the fuzzy Gaussian rule system still supplies its gating weights.

Temporal GRU example:

```yaml
- name: dnfs
  params:
    encoder_type: gru
    encoder_hidden_size: 64
    encoder_num_layers: 1
    encoder_dropout: 0.1
    latent_size: 32
    num_rules: 16
    strength_reduction: mean
    rule_temperature: 1.0
    min_width: 0.05
    max_width: 5.0
    consequent_type: first_order
    antecedent_dropout: 0.0
    consequent_dropout: 0.1
    rule_initialization: kmeans
    usage_regularization: 0.001
    gating: topk
    top_k_rules: 4
```

K-means initialization runs the initialized encoder over training windows only, then initializes
centers and widths from deterministic latent clusters. Validation and test windows are never used.
Usage regularization is disabled at `0.0`; positive values softly penalize collapse through the KL
divergence of average rule use from uniform use. It does not force exactly uniform per-sample rules.

`residual_mode` can be `none`, `persistence`, or `linear`. Persistence adds the last known price for
price targets and zero for return targets. Linear fits and freezes the same OLS-style flattened
window baseline using that fold's training loader only. In both cases, normalized fuzzy-rule
aggregation remains the learned correction.

Each run saves normalized rule activations, raw strengths, rule consequent outputs, entropy,
dominance, and unused-rule counts by forecast date. The Python API also exposes
`estimate_average_rule_usage`, `identify_unused_rules`, and `prune_rules`; pruning is never automatic.

### DNFS migration

Direct Python construction defaults to the temporal GRU encoder. For reproducibility, YAML files
created before the temporal DNFS that omit `encoder_type` are parsed as:

```yaml
encoder_type: flatten
legacy_mode: true
```

Set `encoder_type: gru` explicitly for the new default architecture. The old `dropout` parameter is
accepted in legacy mode and maps to the historical shared dropped representation. New configs
should use separate `encoder_dropout`, `antecedent_dropout`, and `consequent_dropout`; antecedent
dropout defaults to zero so fuzzy matching is deterministic unless explicitly changed.

The current matched WIG20 ablation supports GRU + first-order consequents + 8 rules with
`usage_regularization: 1e-3` as the research default. It compares five deterministic seeds,
neighboring regularization coefficients, and 4/8/16 rules. See the exact design, limitations,
curated tables, and plots in
[`results/wig20/dnfs_ablation/`](results/wig20/dnfs_ablation/). Regenerate the summaries after
running the tracked study configs with:

```bash
uv run python scripts/analyze_dnfs_ablation.py
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

DNFS tuning uses the same command and additionally searches temporal encoder, latent size, rule
count, consequent family, temperature, width floor, consequent dropout, usage regularization, and
softmax/top-k gating parameters.

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

## Batch experiments

Batch experiments run independently resumable, single-model configurations and aggregate their
results across markets, feature sets, and seeds. The canonical matched-origin study contains 180
runs:

- Markets: WIG20, S&P 500, and BIST100.
- Models: persistence, linear regression, CNN1D, MLP, LSTM-GRU, and DNFS.
- Features: Close only and Close with SMA/RSI/MACD.
- Seeds: `1`, `7`, `42`, `123`, and `2026`.
- Expanding walk-forward evaluation with recursive horizon-20 forecasting.

Technical-indicator warm-up can otherwise shift split boundaries. The canonical batch computes the
maximum required warm-up for each market, applies the same total prefix exclusion to both feature
conditions, materializes one dated fold plan per market, and validates every run against that plan.

Resolve the complete matrix and plans without training:

```bash
uv run forecastle batch \
  --config configs/batches/markets_matched_origins_recursive_h20.yaml \
  --dry-run
```

Run or resume the canonical study:

```bash
uv sync --frozen --dev
uv run forecastle batch --config configs/batches/markets_matched_origins_recursive_h20.yaml
```

The command is resume-safe. Successful runs are validated and skipped. Interrupted or incomplete
runs are attempted again. A failed run whose config, dataset, matched plan, and source revision are
unchanged remains recorded as failed; pass `--retry-failed` to attempt it again. Use `--limit 1` for
a quick orchestration check. Run IDs are deterministic and include every varied batch dimension,
for example
`wig20__cnn1d__close__seed42`.

The stable output directory is
`outputs/batches/markets_matched_origins_recursive_h20/`:

```text
batch_config.yaml
study_metadata.yaml
planned_runs.csv
matched_origins/<market>_plan.csv
matched_origins/<market>_usable_dates.csv
runs/<stable-run-id>/config.yaml
runs/<stable-run-id>/metadata.yaml
runs/<stable-run-id>/artifacts/<timestamp>/...
manifest.csv
matched_origin_integrity.csv
completion_summary.csv
divergence_frequency.csv
run_results.csv
aggregate_metrics.csv
model_rankings.csv
indicator_effects.csv
cross_market_comparison.csv
aggregate_horizon_metrics.csv
aggregate_fold_metrics.csv
seed_stability.csv
plots/
```

Every CSV summary also has a Markdown counterpart. Run metadata records status, exact config hash,
dataset hash, Git revision, package versions, timing, and the selected artifact directory. Study
plots cover model ranking, indicator effects, cross-market performance, per-horizon performance,
and seed stability.

The integrity report checks dated fold and prediction keys for every successful run and compares
persistence predictions and metrics exactly across feature conditions. Any checked mismatch aborts
the canonical batch. Failed combinations remain in manifests, aggregate tables, and rankings with
explicit completion and divergence fields. Rankings use price RMSE, never return MAPE.

The earlier `markets_indicators_recursive_h20` batch is retained as
[curated exploratory historical evidence](results/experiments/markets_indicators_recursive_h20_exploratory/README.md).
Its Close and indicator origins were shifted by indicator warm-up, so its indicator-effect table is
not a controlled ablation and is not reused by the canonical study. The committed subset includes
the study definition, manifest, aggregate and horizon tables, seed summaries, and key plots, while
excluding checkpoints and raw per-run output directories.

The completed canonical study is available under
[`results/experiments/markets_matched_origins_recursive_h20/`](results/experiments/markets_matched_origins_recursive_h20/).
It preserves the matched plans and integrity report, all run outcomes, aggregate/seed/fold/horizon
tables, divergence records, and key plots. The principal result is market-dependent: persistence
remains strongest on WIG20 and BIST100, while technical-indicator CNN1D beats persistence on the
S&P 500 in all five seeds and at all 20 recursive horizons.

### Direct versus recursive and rolling versus expanding

Two follow-up batches reuse the canonical dated forecast schedule committed with the recursive
expanding study:

- `markets_matched_origins_direct_h20` compares direct endpoint forecasts at `t+20` with the
  canonical recursive forecasts at `horizon_step=20`.
- `markets_matched_origins_rolling_recursive_h20` compares a rolling training window with the
  canonical expanding window while retaining recursive steps 1 through 20.

Each batch contains the same 180 market/model/feature/seed combinations as the canonical study.
The source plans contribute forecast origins, target dates, and horizons only. Forecastle derives
new strategy-specific fold boundaries, so direct training uses horizon-20 samples and rolling
training uses its configured fixed historical window.

Resolve both matrices without training:

```bash
uv run forecastle batch \
  --config configs/batches/markets_matched_origins_direct_h20.yaml \
  --dry-run

uv run forecastle batch \
  --config configs/batches/markets_matched_origins_rolling_recursive_h20.yaml \
  --dry-run
```

Run or resume them:

```bash
uv run forecastle batch \
  --config configs/batches/markets_matched_origins_direct_h20.yaml

uv run forecastle batch \
  --config configs/batches/markets_matched_origins_rolling_recursive_h20.yaml
```

After a batch completes, generate paired reports:

```bash
uv run forecastle compare \
  --config configs/comparisons/direct_vs_recursive_h20.yaml

uv run forecastle compare \
  --config configs/comparisons/rolling_vs_expanding_recursive_h20.yaml
```

The direct comparison pairs only direct `t+20` with recursive `horizon_step=20`; it never implies
that direct forecasts exist for steps 1 through 19. The rolling comparison also writes paired
per-fold and per-horizon summaries. Both comparison commands verify dated schedules and persistence
controls before writing RMSE/MAE deltas, seed win counts, coverage tables, Markdown summaries, and
plots under `outputs/comparisons/`. Return MAPE is not used for ranking.

The completed follow-up studies are curated under
[`results/experiments/markets_matched_origins_direct_h20/`](results/experiments/markets_matched_origins_direct_h20/),
[`results/experiments/markets_matched_origins_rolling_recursive_h20/`](results/experiments/markets_matched_origins_rolling_recursive_h20/),
and [`results/comparisons/`](results/comparisons/). Direct forecasting is not uniformly better than
recursive step 20, and rolling windows are not uniformly better than expanding windows. Persistence
still leads WIG20 and BIST100; S&P 500 indicator CNN1D remains the strongest repeatable learned
result.

Forecastle uses Matplotlib's non-interactive `Agg` backend for file artifacts, including when
Kaggle or Colab exports a notebook backend that is unavailable inside the isolated `uv` environment.

### Kaggle

In a Kaggle notebook with internet access, use a persistent working directory and let `uv` provide
Python 3.12 when the notebook image does not already have it:

```python
!git clone https://github.com/ahmetesad/forecastle.git
%cd forecastle
!pip install -q uv
!uv python install 3.12
!uv sync --frozen --python 3.12
!uv run forecastle batch --config configs/batches/markets_matched_origins_recursive_h20.yaml
```

Kaggle sessions are finite, so archive or save the stable batch directory as a notebook output.
Restoring that directory before rerunning preserves resume state.

### Colab

The same workflow runs in Colab:

```python
!git clone https://github.com/ahmetesad/forecastle.git
%cd forecastle
!pip install -q uv
!uv python install 3.12
!uv sync --frozen --python 3.12
!uv run forecastle batch --config configs/batches/markets_matched_origins_recursive_h20.yaml
```

For runs that span multiple Colab sessions, copy
`outputs/batches/markets_matched_origins_recursive_h20/` to mounted Google Drive at the end of a session
and restore it to the same path before resuming.
