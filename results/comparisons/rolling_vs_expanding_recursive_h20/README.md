# Rolling Versus Expanding Recursive Evaluation

This comparison holds forecast origins, target dates, horizons, feature conditions, models, and
seeds fixed while changing only the historical training window. Negative deltas in the comparison
tables mean rolling-window evaluation has lower error.

All dated schedules and persistence controls pass exactly. The comparison contains 175 paired
runs; BIST100 indicator linear regression is absent because all five rolling and all five
canonical expanding runs diverged.

## Main results

- There is no universal rolling-window advantage.
- WIG20 generally favors expanding history. Rolling raises aggregate price RMSE by about `3.39%`
  for Close-only CNN1D, `4.40%` for MLP, `4.51%` for LSTM-GRU, and `12.38%` for linear regression.
- S&P 500 window effects are mostly small. Rolling indicator CNN1D is `2.31%` worse, while
  indicator LSTM-GRU is `0.37%` better.
- BIST100 rolling windows help selected models: indicator DNFS improves by `3.18%`, indicator
  LSTM-GRU by `2.51%`, and Close-only linear regression by `9.75%`. They substantially hurt MLP
  and CNN1D.
- Horizon effects reinforce the market dependence. At step 20, rolling improves BIST100 indicator
  DNFS by `5.12%` and LSTM-GRU by `3.10%`, but worsens WIG20 Close-only CNN1D by `5.48%` and
  LSTM-GRU by `6.34%`.

Expanding windows remain the safer general default. Rolling windows are useful as a
market/model-specific robustness check rather than a consistently superior protocol.
