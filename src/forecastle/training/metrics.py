from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape:
        msg = "y_true and y_pred must have the same shape."
        raise ValueError(msg)

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    nonzero = np.abs(actual) > np.finfo(np.float64).eps
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100.0)
        if nonzero.any()
        else float("nan")
    )
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = 1.0 - float(np.sum(np.square(errors))) / denominator if denominator > 0.0 else float("nan")
    return RegressionMetrics(mae=mae, rmse=rmse, mape=mape, r2=r2)
