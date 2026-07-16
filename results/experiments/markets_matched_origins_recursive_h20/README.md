# Canonical Matched-Origin Three-Market Batch

This directory contains the curated results of the canonical three-market recursive forecasting
study. Unlike the earlier exploratory batch, both feature conditions use identical usable dates,
walk-forward folds, forecast origins, target dates, horizons, and prediction counts within each
market.

## Study Design

- Source revision: `35d78471bb69131d11d7a802603232a9a20a0523`
- Markets: WIG20, S&P 500, and BIST100
- Models: naive persistence, linear regression, CNN1D, MLP, LSTM-GRU, and DNFS
- Feature conditions: Close only; Close with SMA/RSI/MACD
- Seeds: `1`, `7`, `42`, `123`, and `2026`
- Evaluation: expanding walk-forward
- Forecasting: recursive 20-step log-return forecasting
- Planned runs: 180
- Completed runs: 175
- Failed runs: 5

DNFS uses the selected canonical configuration: GRU encoder, first-order consequents, eight rules,
K-means rule initialization, and `usage_regularization=1e-3`.

## Integrity

All 175 completed runs passed the canonical-plan validation:

- Forecast origins and target dates match.
- Fold boundaries and fold counts match.
- Prediction counts and horizons match.
- All 15 market/seed persistence pairs have identical predictions and metrics across feature
  conditions.

The study therefore provides a controlled comparison of Close-only inputs against generated
technical indicators. The detailed evidence is in `matched_origin_integrity.csv`.

## Aggregate Results

| Market | Close winner | Indicator winner | Persistence price RMSE |
| --- | --- | --- | ---: |
| WIG20 | naive persistence, 96.584 | naive persistence, 96.584 | 96.584 |
| S&P 500 | linear regression, 163.811 | CNN1D, 153.870 | 167.262 |
| BIST100 | naive persistence, 544.332 | naive persistence, 544.332 | 544.332 |

### WIG20

Persistence remains the strongest model in both feature conditions. The closest model is LSTM-GRU:
2.36% worse with Close only and 2.48% worse with indicators. No learned indicator model beats
persistence in any seed. Persistence is best from recursive horizon 10 through 20 in the indicator
condition.

Technical indicators do not provide a meaningful aggregate improvement. They make CNN1D 0.57%
worse, LSTM-GRU 0.12% worse, and MLP 3.10% worse. Linear regression improves by 0.73%, but remains
5.92% behind persistence. The earlier isolated CNN1D win is therefore not seed-stable.

### S&P 500

The S&P 500 provides the strongest learned-model evidence:

- Close-only linear regression beats persistence by 2.06% in all five nominal seed runs.
- Close-only LSTM-GRU beats persistence by 1.16% on average and in all five seeds.
- Indicator CNN1D beats persistence by 8.01% on average and in every seed.
- Indicator CNN1D is best at every recursive horizon from 1 through 20.
- At horizon 20, indicator CNN1D remains 7.40% better than persistence.
- Indicators improve CNN1D by 9.15% on average, with improvements in all five paired seeds.

The CNN1D advantage is not universal across periods. It beats persistence in 58% of individual
seed-fold comparisons, and the five most favorable folds provide approximately 62% of its gross
squared-error reduction. The result is robust across seeds and horizons, but partly concentrated
in favorable market regimes.

### BIST100

Persistence remains strongest in both feature conditions. Close-only LSTM-GRU is the nearest
challenger at 2.71% worse on average, despite beating persistence in two of five seeds. Indicator
DNFS is 4.21% worse and indicator LSTM-GRU is 4.51% worse. Persistence wins 16 of 20 Close-only
horizons and 18 of 20 indicator horizons.

Indicators have mixed effects but do not create a competitive learned model. DNFS price RMSE
improves by 1.80% relative to its Close-only result, while its return RMSE becomes slightly worse.

## Numerical Stability

All five failures are the BIST100 indicator linear-regression condition. Each nominal seed reaches
the same deterministic divergence:

- Fold: 45
- Forecast origin: `2026-02-27`
- Horizon step: 12
- Previous recursively reconstructed price: approximately 6.62 billion
- Predicted log return: approximately -541,505
- Reconstructed price: 0

No clipping, bounding, or fallback prediction was applied. The failed condition remains visible in
the manifest, completion summary, divergence summary, and rankings. The canonical DNFS
configuration completed all runs, including BIST100 with indicators.

## Interpretation

Technical indicators are market-dependent rather than universally useful. They provide a strong,
seed-stable benefit for CNN1D on the S&P 500, little or negative value on WIG20, and mixed but
noncompetitive changes on BIST100.

The broader finding is also market-dependent:

- WIG20: persistence wins.
- BIST100: persistence wins.
- S&P 500: simple linear regression and LSTM-GRU beat persistence with Close alone, while indicator
  CNN1D wins convincingly.

Return MAPE is retained only as a diagnostic and is not used for ranking.

## Curated Artifacts

- `batch_config.yaml` and `study_metadata.yaml`: exact study definition and provenance.
- `planned_runs.csv`, `manifest.csv`, and `completion_summary.*`: complete run accounting.
- `matched_origin_integrity.*` and `matched_origins/`: controlled-ablation evidence.
- `run_results.csv`: metrics and timing for completed runs.
- `aggregate_metrics.*`, `model_rankings.*`, and `cross_market_comparison.*`: aggregate results.
- `indicator_effects.*` and `indicator_effect_summary.*`: paired indicator effects.
- `aggregate_horizon_metrics.*` and `horizon_results.csv`: recursive-horizon behavior.
- `aggregate_fold_metrics.*` and `fold_results.csv`: fold-level stability.
- `seed_stability.*`: seed variation.
- `divergence_frequency.*`: explicit numerical-failure accounting.
- `plots/`: generated high-level figures.

Timestamped run directories, checkpoints, raw per-run predictions, and large duplicate Markdown
tables are intentionally omitted. They can be regenerated from the tracked configuration, source
revision, matched plans, and committed dataset snapshots.
