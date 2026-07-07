# WIG20 LSTM Tuning Summary

Tuning run: `outputs/wig20_lstm_tuning/20260707_155330`

Held-out test run: `outputs/wig20_lstm_tuned/20260707_155818`

The Optuna study tuned an LSTM on WIG20 OHLCV log-return forecasting using validation
price RMSE as the objective.

## Search Space

- `sequence_length`: `7`, `14`, `30`, `60`, `120`
- `batch_size`: `16`, `32`, `64`, `128`
- `hidden_size`: `16`, `32`, `64`, `128`, `256`
- `num_layers`: `1` to `3`
- `dropout`: `0.0` to `0.5`
- `learning_rate`: log scale from `1e-5` to `1e-2`
- `weight_decay`: log scale from `1e-8` to `1e-2`

## Best Validation Trial

Trial `14` was best.

| Parameter | Value |
| --- | --- |
| `sequence_length` | 7 |
| `batch_size` | 32 |
| `hidden_size` | 16 |
| `num_layers` | 1 |
| `dropout` | 0.4949525716 |
| `learning_rate` | 0.0030817614 |
| `weight_decay` | 0.0004294939 |

Validation result:

| Model | Validation Price RMSE | Validation Return RMSE |
| --- | ---: | ---: |
| Tuned LSTM | 28.7256 | 0.0152510 |
| Persistence | 28.9290 | 0.0153828 |

The tuned LSTM improved validation price RMSE by about `0.7%`.

## Held-Out Test Result

| Model | Test Price RMSE | Test Return RMSE |
| --- | ---: | ---: |
| Naive persistence | 35.1429 | 0.0131344 |
| Tuned LSTM | 35.4736 | 0.0132411 |
| Linear regression | 35.7003 | 0.0132857 |

The validation gain did not transfer to the held-out test period. Persistence remained the
strongest out-of-sample benchmark.

## Interpretation

Optuna found a validation-improving LSTM configuration, but the result did not generalize to
the test split. This strengthens the research conclusion: even after architecture comparison
and LSTM hyperparameter optimization, naive persistence remains difficult to beat on WIG20
with historical OHLCV inputs alone.
