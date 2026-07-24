# G7 Expanding Recursive Benchmark

This directory contains the curated evidence from the controlled seven-market benchmark using
expanding walk-forward evaluation and recursive 20-step log-return forecasting.

## Study Design

- Model-run revision: `2ba135f24dbe533e013485e62c3f128777042146`
- Combined-report revision: `4afc3059a3c1e8ab0108df830c257a325480a7d4`
- Calendar bounds: 2006-01-04 through 2026-07-22
- Markets: Canada, France, Germany, Italy, Japan, the United Kingdom, and the United States
- Models: naive persistence, linear regression, CNN1D, LSTM-GRU, and DNFS
- Features: Close only; Close with SMA, RSI, and MACD
- Neural seeds: `1`, `42`, and `2026`
- Deterministic baseline seed: `42`
- Planned and completed runs: 154
- Failed or divergent runs: 0

Each market retains its native trading sessions. Close-only and indicator conditions use the same
33-row warm-up, usable dates, fold boundaries, forecast origins, target dates, and horizons. The
DAX source is Yahoo's `^GDAXI` performance index and therefore incorporates distributions.

Standalone LSTM was not included in this model set. These results cannot determine whether
LSTM-GRU improves on an ordinary LSTM across the G7 markets.

## Integrity

All 154 runs passed their canonical-plan checks. Baseline and neural slices use identical dataset
and matched-plan hashes for each market. Persistence predictions and metrics are identical between
the two feature conditions in every market.

The deterministic persistence control is stored once per market and feature condition. Combined
reporting broadcasts that control across neural seeds when calculating ratios, seed-win counts,
and ranks; it does not create duplicate baseline runs.

## Aggregate Price-RMSE Results

| Market | Best model and features | Difference from persistence | Seeds beating persistence |
| --- | --- | ---: | ---: |
| Canada | CNN1D with indicators | 6.87% better | 3/3 |
| France | naive persistence | 0.00% | n/a |
| Germany | DNFS with indicators | 0.71% better | 2/3 |
| Italy | naive persistence | 0.00% | n/a |
| Japan | CNN1D with indicators | 2.31% better | 2/3 |
| United Kingdom | naive persistence | 0.00% | n/a |
| United States | Close-only LSTM-GRU | 5.28% better | 3/3 |

Germany is best interpreted as a tie: the mean advantage is below 1% and DNFS varies substantially
across seeds. France, Italy, and the United Kingdom provide clear evidence in favor of persistence.
Canada, Japan, and the United States provide the strongest learned-model evidence.

## Feature Effects

Technical indicators are architecture- and market-dependent:

- DNFS mean price RMSE improves with indicators in all seven markets, although it does not always
  become competitive with persistence.
- CNN1D receives large indicator gains in Canada and the United Kingdom. The United Kingdom model
  still remains behind persistence.
- Indicators generally worsen LSTM-GRU; the United Kingdom is the exception.
- Indicators generally worsen linear regression.

No single feature condition should therefore be treated as universally preferable.

## Horizon and Fold Stability

Canada Close-only LSTM-GRU and indicator DNFS beat persistence at all 20 recursive horizons.
United States Close-only LSTM-GRU and CNN1D also beat persistence at all 20 horizons. Japan's
CNN1D advantage is strongest at longer horizons.

The United States Close-only LSTM-GRU result is the strongest across folds: it beats persistence
in 25 of 39 folds, with a median fold improvement of 6.86%. Japan indicator CNN1D wins 23 of 38
folds. Canada indicator CNN1D wins 20 of 39 folds; its aggregate advantage is meaningful but partly
weighted toward higher-error periods.

Return MAPE is retained only as a diagnostic and is not used for ranking. Return R-squared is often
negative, while price R-squared is inflated by trending index levels. Price RMSE, price MAE, and
matched improvement over persistence are the primary comparisons.

## Curated Artifacts

- `manifest.csv`, `completion_summary.*`, and `divergence_frequency.*`: complete run accounting.
- `aggregate_metrics.*`, `model_rankings.*`, and `run_results.csv`: combined model comparisons.
- `indicator_effects.csv` and `indicator_effect_summary.*`: paired feature effects.
- `aggregate_horizon_metrics.csv`: mean and standard deviation by recursive horizon.
- `aggregate_fold_metrics.csv`: fold-level stability across seeds.
- `seed_stability.csv`: seed variation by market, feature condition, and model.
- `integrity/`: baseline and neural canonical-plan validation.
- `matched_origins/`: exact dated forecast schedules for all seven markets.
- `study_metadata_*.yaml`: source configuration hashes, package provenance, and run definitions.
- `plots/`: model ranking, feature effect, cross-market, horizon, and seed-stability figures.

Checkpoints, queue logs, timestamped run directories, and raw per-run predictions are intentionally
omitted. The committed configurations and fixed dataset snapshots reproduce those artifacts.
