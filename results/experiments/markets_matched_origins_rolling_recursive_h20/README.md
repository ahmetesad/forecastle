# Rolling Recursive Horizon-20 Three-Market Study

This study replaces the canonical expanding training window with a fixed rolling window while
retaining exactly the same forecast origins, targets, recursive horizons, feature conditions,
models, and five deterministic seeds.

The batch completed 175 of 180 runs. All five BIST100 indicator linear-regression runs failed
deterministically at fold 20, horizon step 19, after recursive reconstruction diverged. The failed
combination remains visible and is excluded from rankings rather than repaired or clipped.

## Main results

- Persistence remains best on WIG20 and BIST100.
- On the S&P 500, Close-only linear regression reaches `162.84` price RMSE versus `167.26` for
  persistence.
- S&P 500 indicator CNN1D is again strongest at `157.40` and beats persistence in all five seeds.
- S&P 500 indicator LSTM-GRU also beats persistence in all five seeds at `163.15`.
- Indicators improve S&P 500 CNN1D by `6.52%`, WIG20 LSTM-GRU by `2.20%`, and BIST100 DNFS by
  `3.14%`.
- Indicators harm BIST100 MLP and CNN1D and remain numerically unsafe for BIST100 linear
  regression under recursive forecasting.

The matched rolling-versus-expanding comparison is under
[`results/comparisons/rolling_vs_expanding_recursive_h20/`](../../comparisons/rolling_vs_expanding_recursive_h20/).

## Curated artifacts

This directory retains the exact batch configuration, study metadata, dated plans, complete
manifest, aggregate and seed-level tables, raw run/fold/horizon CSVs, integrity and divergence
reports, and key plots. Per-run checkpoints and generated artifact trees are intentionally
excluded.
