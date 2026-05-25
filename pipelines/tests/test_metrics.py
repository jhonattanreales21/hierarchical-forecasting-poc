"""Unit tests for shared forecast evaluation metrics."""

import warnings

import numpy as np
from shared.metrics import mape, mase, rmse, wape


def test_mape_basic():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    result = mape(y_true, y_pred)
    assert isinstance(result, float)
    assert result > 0
    assert abs(result - 0.0611) < 0.001  # noqa: PLR2004


def test_mape_excludes_zeros():
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 190.0])
    result = mape(y_true, y_pred)
    # Zero in y_true is excluded; only two values contribute
    assert isinstance(result, float)
    assert not np.isnan(result)


def test_mape_all_zeros_returns_nan():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = mape(y_true, y_pred)
    assert np.isnan(result)


def test_rmse_basic():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    result = rmse(y_true, y_pred)
    assert isinstance(result, float)
    assert abs(result - 10.0) < 1e-6  # noqa: PLR2004


def test_rmse_perfect_forecast():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == 0.0


def test_mase_basic():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    y_train = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    result = mase(y_true, y_pred, y_train, seasonality=1)
    assert isinstance(result, float)
    assert result > 0


def test_mase_short_train_returns_nan_not_raises():
    """When training history is too short for MASE, return NaN with a warning."""
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    y_train = np.array([80.0])  # length 1, seasonality 1 → no valid naive pairs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mase(y_true, y_pred, y_train, seasonality=1)
    assert np.isnan(result), "Expected NaN when training history is too short for MASE"
    assert len(caught) >= 1, "Expected at least one warning for short training history"


# ── WAPE tests ────────────────────────────────────────────────────────────────


def test_wape_basic():
    """WAPE uses sum-based formula, not a mean of individual errors."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    # |100-110| + |200-190| + |300-310| = 10 + 10 + 10 = 30
    # sum(|y_true|) = 600
    # WAPE = 30 / 600 = 0.05
    result = wape(y_true, y_pred)
    assert isinstance(result, float)
    assert abs(result - 0.05) < 1e-9  # noqa: PLR2004


def test_wape_differs_from_mean_mape():
    """WAPE aggregates by sum, so it weights large observations more than MAPE."""
    y_true = np.array([1.0, 1000.0])
    y_pred = np.array([2.0, 1100.0])
    result = wape(y_true, y_pred)
    # |1-2| + |1000-1100| = 1 + 100 = 101
    # sum(|y_true|) = 1001
    # WAPE ≈ 0.1009 — dominated by the large value
    assert abs(result - (101 / 1001)) < 1e-9  # noqa: PLR2004


def test_wape_perfect_forecast():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert wape(y_true, y_pred) == 0.0


def test_wape_zero_denominator_returns_nan():
    """WAPE returns NaN when all actuals are zero (avoid division by zero)."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = wape(y_true, y_pred)
    assert np.isnan(result)
