# Forecastle Project Status

Last updated: 2026-07-15

## Summary

Current experimental evidence on WIG20 indicates that naive persistence remains the strongest overall baseline under the canonical protocol, although several neural architectures approach its performance. Remaining work focuses on additional datasets, multi-seed validation, and statistical analysis rather than major framework development.

## Current implementation

The repository currently has:

- Neural models: MLP, RNN, LSTM, GRU, CNN1D, DNFS, LSTM-GRU, and CNN-LSTM.
- Baselines: naive persistence and ordinary least-squares linear regression.
- Chronological holdout evaluation with leakage-safe, training-only scaling.
- Expanding and rolling walk-forward evaluation with fresh fold models and deterministic seeds.
- Direct endpoint forecasting and recursive multi-step forecasting.
- Price, simple-return, and log-return targets with price reconstruction.
- Causal SMA, RSI, and MACD features with indicator warm-up removal.
- Aggregate, per-fold, and per-horizon metrics, long-form predictions, plots, and checkpoints.
- YAML experiment definitions, parameter sweeps, and resumable Optuna tuning.
- Stable-ID, resumable batch experiments with cross-run summaries and plots.
- Temporal DNFS encoders, Gaussian fuzzy rules, configurable TSK consequents, training-only
  K-means initialization, rule diagnostics, usage regularization, top-k gating, and optional pruning.
- Yahoo Finance downloads for WIG20, BIST100, and S&P 500 presets.

Existing holdout configurations keep their original behavior. Walk-forward evaluation, recursive forecasting, and technical indicators are enabled only when their configuration sections request them. Recursive forecasting intentionally accepts only Close and Close-derived indicators because future OHLCV or exogenous observations are unavailable at inference time.

## Completed experiments

### WIG20 feature/lookback/horizon sweep

The initial holdout sweep compared Close and OHLCV features, lookbacks of 7, 14, 30, 60, and 120,
and horizons of 1, 5, 10, and 20. Persistence won return RMSE in all 40 configurations and price
RMSE in 39 of 40; the remaining result was a numerical tie-level LSTM improvement. OHLCV did not
consistently improve learned-model performance. See
[`results/wig20/sweep_summary.md`](results/wig20/sweep_summary.md).

### WIG20 LSTM tuning

Optuna found an LSTM that improved validation price RMSE by about 0.7%, but the gain did not transfer
to the held-out test period. Persistence remained strongest on the held-out data. See
[`results/wig20/lstm_tuning_summary.md`](results/wig20/lstm_tuning_summary.md).

### Canonical WIG20 walk-forward run

The canonical post-randomness-fix run uses expanding walk-forward evaluation, recursive 20-step
log-return forecasts, causal technical indicators, seed 42, 38 folds, and all supported models.
Persistence ranks first in aggregate reconstructed-price RMSE and return RMSE:

| Result | Model | RMSE | Difference from persistence |
| --- | --- | ---: | ---: |
| Reconstructed price | naive_persistence | 96.5845 | 0.00% |
| Best learned price model | cnn_lstm | 98.9378 | 2.44% worse |
| Return | naive_persistence | 0.037457 | 0.00% |
| Best learned return model | lstm_gru | 0.038276 | 2.19% worse |

Linear regression is best at recursive step 1, MLP at steps 2-4 and 6-8, CNN1D at step 5, and
persistence at steps 9-20. CNN-LSTM is the closest learned model in aggregate price space, but this
single-seed result is not evidence that its ranking is seed-stable. The curated config, metadata,
tables, plots, integrity report, and representative predictions are under
[`results/wig20/walk_forward_recursive/`](results/wig20/walk_forward_recursive/).

Return-space MAPE is retained only as a diagnostic because percentage errors become unstable when
actual returns approach zero. It is not used to rank models. Negative R-squared values are possible
and should be read as performance below a constant-mean predictor, not as an implementation error.

### WIG20 temporal DNFS ablation

A matched five-seed study compared usage-regularization coefficients and rule counts for a GRU
DNFS with first-order TSK consequents. `usage_regularization=1e-3` reduced mean price RMSE from
103.893 to 100.785, improved four of five paired seeds, and reduced seed standard deviation from
7.971 to 5.760. Its gains were strongest at recursive horizons 5-20; rule balance itself had only a
weak relationship with accuracy.

At `1e-3`, 8 rules outperformed 4 rules in four of five seeds and at horizons 2-20. Sixteen rules
lost to 8 in every seed and left about nine rules below 1% mean usage. The current DNFS default is
therefore GRU + first-order consequents + 8 rules + `usage_regularization=1e-3`. Full design notes,
curated tables, and plots are under
[`results/wig20/dnfs_ablation/`](results/wig20/dnfs_ablation/).

### Exploratory three-market recursive batch

The first full batch attempted 180 combinations across WIG20, S&P 500, BIST100, six models, two
feature conditions, and five seeds. It completed 170 runs; five BIST100 linear-regression indicator
runs and five BIST100 DNFS indicator runs failed after recursive forecasts produced non-finite
histories. Within individual feature conditions, the strongest learned-model evidence appeared on
the S&P 500, while persistence remained strongest on BIST100 and close to the best WIG20 result.

This batch is evidence of completed experimentation, but it is not a controlled indicator
ablation: indicator warm-up shifted the usable dates and forecast origins relative to Close-only
runs. Its generated indicator-effect tables are therefore retained but explicitly marked as
confounded. Curated configs, manifests, summaries, horizon tables, and plots are under
[`results/experiments/markets_indicators_recursive_h20_exploratory/`](results/experiments/markets_indicators_recursive_h20_exploratory/).

## Research artifact policy

`outputs/`, checkpoints, Optuna databases, and temporary runs remain ignored. The three datasets
used by tracked experiments are committed as fixed research snapshots to simplify collaboration;
additional raw downloads remain ignored. The repository otherwise keeps selected final and
historically important exploratory tables, key plots, exact experiment configs, concise analyses,
checksums, manifests, and representative prediction files. Exploratory artifacts are labeled with
their methodological limitations. This makes the research trail inspectable while avoiding large
or easily regenerated artifacts.

## Known limitations

- The canonical walk-forward result currently uses one deterministic seed.
- The DNFS ablation uses five seeds but only five walk-forward origins and one market.
- Recursive forecasting supports only Close and generated Close-derived indicators.
- Direct forecasting predicts one endpoint at `t+h`; it is not a multi-output sequence decoder.
- Hybrid-specific Optuna search spaces are not implemented.
- Walk-forward training is sequential and can be slow across many models and folds.
- Committed market snapshots are static and must be refreshed deliberately when extending the study.
- The current evidence does not isolate indicator effects from recursive-vs-direct strategy effects.
- Earlier exploratory ablation and seed runs preceded the model-order randomness fix and are not
  treated as canonical results.

## Remaining work

1. Rerun the matched WIG20 indicator and direct-vs-recursive ablation from the fixed implementation.
2. Run at least three post-fix deterministic seeds for persistence and the strongest non-DNFS challengers.
3. Run full rolling-window evaluation and compare rankings with expanding windows.
4. Reproduce the canonical protocol on BIST100 and S&P 500 and test parameter transfer.
5. Add uncertainty estimates or statistical tests across folds and seeds.
6. Consider parallel fold/trial execution after reproducibility guarantees are preserved.
7. Run and curate the configured three-market, two-feature, five-seed batch study.
8. Replicate the recommended DNFS configuration on BIST100 and S&P 500 with more walk-forward origins.

## Useful commands

```bash
uv sync --dev # installing dependencies (if you have uv installed)
uv run forecastle run --config configs/evaluation/wig20_walk_forward_recursive.yaml
uv run forecastle run --config configs/evaluation/wig20_rolling_recursive.yaml
uv run forecastle run --config configs/evaluation/wig20_walk_forward_recursive_smoke.yaml

# testing
uv run ruff format --check .
uv run ruff check .
uv run pytest
```
