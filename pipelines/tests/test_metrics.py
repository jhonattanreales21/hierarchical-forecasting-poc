"""Unit tests for shared forecast evaluation metrics."""

import numpy as np
import pytest

from shared.metrics import mape, mase, rmse


def test_mape_basic():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    result = mape(y_true, y_pred)
    assert isinstance(result, float)
    assert result > 0
    assert abs(result - 0.0611) < 0.001


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
    assert abs(result - 10.0) < 1e-6


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


def test_mase_raises_when_train_too_short():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    y_train = np.array([80.0])
    with pytest.raises(ValueError):
        mase(y_true, y_pred, y_train, seasonality=1)
