# Direct Versus Recursive at Horizon 20

This comparison pairs direct `t+20` predictions only with recursive `horizon_step=20` predictions
from the canonical expanding study. It does not treat direct forecasts as though they existed for
steps 1 through 19.

All dated schedules pass. Persistence price metrics are exactly identical. Target-space
persistence metrics differ by at most `9.9e-9` because direct log-return calculation and recursive
cumulative reconstruction take slightly different floating-point paths; this is below the
documented `1e-7` comparison tolerance.

## Main results

- Direct forecasting is not uniformly better than recursive forecasting.
- On BIST100 Close-only data, direct substantially stabilizes linear regression (`66.83%` lower
  price RMSE) and modestly improves DNFS (`5.90%` lower).
- Direct BIST100 CNN1D and MLP are much worse than their recursive step-20 counterparts, showing
  that removing feedback does not automatically make the endpoint task easier.
- On the S&P 500, most differences are small. Direct indicator CNN1D improves by `1.35%`, while
  several other models change by only a few percent.
- On WIG20, direct Close-only DNFS improves by `4.20%` and direct indicator MLP by `4.40%`, but
  persistence remains stronger than every learned model.
- BIST100 indicator linear regression has no paired estimate because the canonical recursive runs
  failed; all direct runs completed.

The evidence supports model-specific recursive error accumulation, especially for BIST100 linear
regression, rather than a universal advantage for direct forecasting.
