# WIG20 Sweep Summary

Run directory: `outputs/sweeps/wig20_features_lookbacks_horizons/20260707_145540`

This sweep evaluated WIG20 log-return forecasting across:

- Feature sets: `Close`, `OHLCV`
- Lookbacks: `7`, `14`, `30`, `60`, `120`
- Horizons: `1`, `5`, `10`, `20`
- Methods: MLP, RNN, LSTM, GRU, 1D CNN, linear regression, naive persistence

## Main Result

Naive persistence was the strongest benchmark.

- Winner by return-space RMSE: `40 / 40` configurations
- Winner by reconstructed price RMSE: `39 / 40` configurations
- The single non-persistence price-RMSE win was an LSTM on `OHLCV`, lookback `120`,
  horizon `1`, and was only a numerical tie-level improvement.

## OHLCV vs Close

Including persistence, OHLCV did not change the best configuration because persistence does
not use input features.

Among non-persistence models:

- OHLCV improved `49 / 120` model-configuration comparisons
- Mean price-RMSE delta: `+0.721%` worse
- Median price-RMSE delta: `+0.368%` worse

OHLCV did not consistently improve performance in this first sweep.

## Lookback and Horizon

The best overall configuration for every horizon used a `7` day lookback, again because
persistence dominated.

Best non-persistence models by horizon:

| Horizon | Model | Feature Set | Lookback | Price RMSE |
| --- | --- | --- | --- | --- |
| 1 | CNN1D | OHLCV | 7 | 35.2091 |
| 5 | LSTM | OHLCV | 30 | 73.8664 |
| 10 | CNN1D | OHLCV | 14 | 102.7008 |
| 20 | MLP | OHLCV | 14 | 138.0013 |

## Interpretation

Historical OHLCV windows alone were not enough to consistently beat the persistence baseline
on WIG20. This is a useful empirical result rather than a failed experiment: it motivates
hyperparameter tuning, cross-market testing, rolling-window evaluation, and additional
feature engineering.
