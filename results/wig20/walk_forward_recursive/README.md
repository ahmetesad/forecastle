# Canonical WIG20 Walk-Forward Results

This is the canonical post-fix run from source commit `e5cbb5fdd6a45099db01275a112953de898ba68c`. It uses
expanding walk-forward evaluation, recursive 20-step log-return forecasting, Close-derived
technical indicators, seed 42, and every supported model plus persistence and linear regression.

## Main result

Naive persistence ranks first in both reconstructed-price RMSE and return-space RMSE.

| Rank type | Model | RMSE | Delta vs persistence |
| --- | --- | ---: | ---: |
| Price | cnn_lstm | 98.9378 | +2.44% |
| Return | lstm_gru | 0.038276 | +2.19% |
| Persistence price | naive_persistence | 96.5845 | 0.00% |
| Persistence return | naive_persistence | 0.037457 | 0.00% |

The run contains 38 forecast origins and 741 forecast rows per model. The final origin has one
observable target day, so horizon steps 1-20 have 38 observations except later steps, which have
37. Return-space MAPE is retained as a diagnostic artifact but is not used for ranking because
returns frequently approach zero.

## Interpretation

The model-order randomness fix changes individual neural results but not the principal finding:
persistence remains the strongest aggregate benchmark. CNN-LSTM is the closest learned model in
price space, while LSTM-GRU is closest in return space. These are single-seed results and should
not be interpreted as evidence of seed-stable architecture superiority.
