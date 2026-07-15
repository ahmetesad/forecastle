# WIG20 DNFS Regularization And Rule-Count Ablation

This matched study evaluates the temporal DNFS with a GRU encoder and first-order
Takagi-Sugeno consequents. Every condition uses the same WIG20 snapshot, causal
SMA/RSI/MACD features, log-return target, 30-day input window, recursive 20-step
forecasting, five expanding walk-forward folds, K-means initialization from each fold's
training windows only, and seeds `1`, `7`, `42`, `123`, and `2026`. Training uses 20 epochs,
patience 4, batch size 64, and learning rate `1e-3`.

The raw `outputs/` runs and checkpoints are intentionally not tracked. Exact configs,
the reproducible analyzer at `scripts/analyze_dnfs_ablation.py`, concise tables, and plots
are retained here.

## Design verification

- All configurations produced the same five forecast origins.
- Persistence price RMSE was identical across every batch and seed.
- K-means initialization receives the training loader only; focused leakage and
  initialization tests passed before the study.
- Completed `usage_regularization=1e-3`, 8-rule runs were reused rather than overwritten.

## Phase 1: usage regularization

| Coefficient | Price RMSE | Seed SD | Price MAE | Improvement vs persistence | Seeds won | Fold win rate | Effective rules |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 103.893 | 7.971 | 79.650 | -0.80% | 3/5 | 68% | 3.63 |
| 1e-4 | 104.412 | 8.402 | 80.711 | -1.31% | 3/5 | 64% | 3.69 |
| 3e-4 | 104.033 | 8.197 | 80.090 | -0.94% | 3/5 | 64% | 3.71 |
| **1e-3** | **100.785** | **5.760** | **78.234** | **+2.21%** | **3/5** | **72%** | **3.79** |
| 3e-3 | 103.356 | 9.415 | 80.480 | -0.28% | 3/5 | 56% | 3.84 |

`1e-3` improves on no regularization in four of five paired seeds and beats every nearby
coefficient in four of five seeds. Its gain is not uniform across folds: it improves the
two latest folds, especially fold 3, while folds 0-2 are unchanged or worse. The clearest
effect is across recursive horizons 5-20, where its mean price RMSE is roughly 2.5-4.2
points below the unregularized model. Horizon win frequency remains near 50%, so the
coefficient primarily reduces the magnitude of bad errors rather than winning every cell.

Increasing the coefficient gradually raises effective rule use from 3.63 to 3.84 of 8.
Even at `3e-3`, normalized entropy is only 0.629 and about 5.28 rules exceed 1% mean usage;
the system is not being forced close to uniform use. Fold-level balance/accuracy
correlations are weak (`0.00-0.31`), so rule balance is not a reliable accuracy proxy.
The instability of `3e-3`, particularly seed 123, argues against stronger regularization.

## Phase 2: rule count

Phase 2 fixes `usage_regularization=1e-3`.

| Rules | Price RMSE | Seed SD | Improvement vs persistence | Seeds won | Effective rules | Effective ratio | Rules below 1% | Parameters |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 102.838 | 6.484 | +0.22% | 2/5 | 2.57 | 0.64 | 0.72 | 16,932 |
| **8** | **100.785** | **5.760** | **+2.21%** | **3/5** | **3.79** | **0.47** | **2.80** | **17,320** |
| 16 | 109.311 | 5.573 | -6.06% | 1/5 | 4.85 | 0.30 | 9.08 | 18,096 |

Eight rules beat four in four of five paired seeds and at horizons 2-20. Four rules are
competitive and use their capacity more efficiently, but they do not match the 8-rule
model's accuracy or stability. Sixteen rules lose to eight in every seed, are worse at
every horizon, and leave about nine rules below 1% usage. Extra rule capacity beyond eight
is therefore unsupported.

Mean training times were 49.25, 40.92, and 39.47 seconds for 4, 8, and 16 rules;
inference times were 0.57, 0.50, and 0.55 seconds. These are accumulated five-fold times,
but early stopping changes epochs across conditions, so they should not be interpreted as
pure throughput benchmarks.

## Recommendation

Use `encoder_type: gru`, `consequent_type: first_order`, `num_rules: 8`, and
`usage_regularization: 1e-3` as the current DNFS research default. This recommendation is
supported by paired seeds, lower variance, long-horizon behavior, and rule-count ablation,
not mean RMSE alone.

The remaining uncertainty is external validity: this study covers one market, five folds,
one encoder/consequent family, and one training budget. The next useful checks are a larger
walk-forward origin set and matched BIST100/S&P 500 replication.
