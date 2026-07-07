from __future__ import annotations

import numpy as np

from forecastle.training.metrics import compute_metrics


def test_compute_metrics_exact_prediction() -> None:
    actual = np.asarray([1.0, 2.0, 3.0])
    predicted = np.asarray([1.0, 2.0, 3.0])

    metrics = compute_metrics(actual, predicted)

    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.mape == 0.0
    assert metrics.r2 == 1.0


def test_compute_metrics_shape_mismatch() -> None:
    actual = np.asarray([1.0, 2.0])
    predicted = np.asarray([1.0])

    try:
        compute_metrics(actual, predicted)
    except ValueError as error:
        assert "same shape" in str(error)
    else:
        raise AssertionError("Expected ValueError for shape mismatch.")
