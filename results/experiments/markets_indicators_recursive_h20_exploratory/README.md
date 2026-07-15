# Exploratory Three-Market Recursive Batch

This directory preserves the curated results of the first three-market, two-feature, five-seed
batch study. It is committed as research provenance: it proves which experiments were attempted,
records successful and failed runs, and preserves the observations that motivated the later
matched-origin methodology.

## Study design

- Source revision: `fd51a670a9a400310af368f6f4907399272b6d50`
- Markets: WIG20, S&P 500, and BIST100
- Models: naive persistence, linear regression, CNN1D, MLP, LSTM-GRU, and DNFS
- Feature conditions: Close only; Close with SMA/RSI/MACD
- Seeds: `1`, `7`, `42`, `123`, and `2026`
- Evaluation: expanding walk-forward
- Forecasting: recursive 20-step log-return forecasts
- Planned runs: 180
- Completed runs: 170
- Failed runs: 10

The ten failures were all BIST100 indicator runs: five linear-regression runs and five DNFS runs.
Their recursive forecasts produced non-finite reconstructed histories. These failures are retained
in `manifest.csv`; no clipping or fallback prediction was applied.

## Critical limitation

**This is not a controlled indicator ablation.** Indicator warm-up removed initial observations
only from the indicator condition. Consequently, Close-only and indicator runs used different
usable date ranges, fold boundaries, forecast origins, target dates, and prediction counts.

Results may be compared between models *within the same market and feature condition*. Values in
`indicator_effects.csv`, `indicator_effect_summary.csv`, and `plots/indicator_effects.png` are
preserved as generated historical artifacts, but they must not be interpreted as causal evidence
that indicators helped or hurt. The canonical replacement is
`configs/batches/markets_matched_origins_recursive_h20.yaml`.

## Exploratory observations

- WIG20 Close-only CNN1D had mean price RMSE `107.622` versus persistence at `108.020`, a small
  average advantage that was not seed-stable enough to establish architecture superiority.
- S&P 500 provided the strongest within-condition learned-model result. Close-only LSTM-GRU had
  mean price RMSE `156.085` versus persistence at `160.200`; under the indicator condition CNN1D
  had `153.870` versus persistence at `167.262`.
- BIST100 persistence ranked first in both successfully completed feature conditions. Recursive
  linear regression and DNFS with indicators also exposed numerical-stability failures.
- Across markets, model usefulness and recursive stability varied more than architecture labels
  alone would suggest. Return MAPE is not used for ranking.

Because origins differ between feature conditions, even persistence has different aggregate
metrics across those conditions. That discrepancy is the clearest evidence of the confound and is
why the matched-origin integrity checks were added.

## Curated artifacts

- `batch_config.yaml` and `study_metadata.yaml`: exact study definition and environment metadata.
- `planned_runs.csv` and `manifest.csv`: complete matrix and durable run outcomes.
- `run_results.csv`: metrics and timing for successful individual runs.
- `aggregate_metrics.*`, `model_rankings.*`, and `cross_market_comparison.*`: aggregate views.
- `aggregate_horizon_metrics.*` and `horizon_results.csv`: recursive-horizon behavior.
- `seed_stability.*`: multi-seed variation.
- `indicator_effects.*` and `indicator_effect_summary.*`: generated but methodologically confounded.
- `plots/`: the five generated high-level figures.

Per-fold checkpoints, raw per-run predictions, timestamped run directories, temporary notebook
files, and duplicate Markdown expansions are intentionally omitted. They can be regenerated from
the tracked configuration, source revision, and committed dataset snapshots.
