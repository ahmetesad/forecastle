# Direct Horizon-20 Three-Market Study

This study evaluates direct endpoint forecasting at `t+20` using the same market dates and
horizon-20 targets as the canonical expanding recursive study. It contains 180 planned runs:
three markets, six models, two matched feature conditions, and five deterministic seeds.

All 180 runs completed. The direct schedule contains 37 complete WIG20 endpoint folds and 49
complete S&P 500 and BIST100 endpoint folds. Final recursive origins that did not have an observed
20th target were intentionally excluded.

## Main results

- Persistence remains best on WIG20 and BIST100 in both feature conditions.
- On the S&P 500, Close-only linear regression is best at `242.95` price RMSE versus `248.55` for
  persistence.
- S&P 500 indicator CNN1D is the strongest direct result at `226.90` price RMSE and beats
  persistence in all five seeds.
- S&P 500 indicator DNFS also beats persistence in all five seeds at `238.60` price RMSE.
- Indicators improve S&P 500 CNN1D by `10.35%` and DNFS by `8.34%` relative to their matched
  Close-only runs.
- Indicator effects remain market- and model-specific. They mildly improve several WIG20 models
  without beating persistence, while strongly degrading BIST100 linear regression and MLP.

The corresponding matched comparison with recursive `horizon_step=20` is under
[`results/comparisons/direct_vs_recursive_h20/`](../../comparisons/direct_vs_recursive_h20/).

## Curated artifacts

This directory retains the exact batch configuration, study metadata, dated plans, complete
manifest, aggregate and seed-level tables, raw run/fold/horizon CSVs, integrity reports, and key
plots. Per-run checkpoints and generated artifact trees are intentionally excluded.
