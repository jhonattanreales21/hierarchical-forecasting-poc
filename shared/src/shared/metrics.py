"""Evaluation metrics for forecast accuracy assessment.

All functions are pure, stateless, and operate on numpy arrays.
They are used both inside Kedro pipeline nodes (evaluation step)
and in backtesting utilities.
"""

import warnings

import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE) — primary business metric.

    Aggregates using sums rather than averaging individual percentage errors,
    which avoids instability when individual y_true values are small or zero.

    Args:
        y_true: Array of actual observed values.
        y_pred: Array of forecasted values, same shape as y_true.

    Returns:
        WAPE as a float in [0, inf). Expressed as a fraction (e.g. 0.15 = 15%).
        Returns NaN when sum(|y_true|) == 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Zero values in y_true are excluded to avoid division by zero.
    Returns a value in [0, inf), expressed as a fraction (not a percentage).

    Args:
        y_true: Array of actual observed values.
        y_pred: Array of forecasted values, same shape as y_true.

    Returns:
        MAPE as a float. Multiply by 100 to get percentage points.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not mask.any():
        return float("nan")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Array of actual observed values.
        y_pred: Array of forecasted values, same shape as y_true.

    Returns:
        RMSE as a float, in the same units as y_true.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int,
) -> float:
    """Mean Absolute Scaled Error.

    Scales the forecast error by the in-sample naive seasonal benchmark,
    defined as the mean absolute error of a seasonal random walk on y_train.

    A MASE < 1 means the model outperforms the naive seasonal benchmark.

    Args:
        y_true: Array of actual values in the evaluation period.
        y_pred: Array of forecasted values, same shape as y_true.
        y_train: Array of historical training values used to compute the
                 naive benchmark scale. Must have length > seasonality.
        seasonality: Seasonal period for the naive benchmark (e.g. 12 for
                     monthly data with annual seasonality, 1 for non-seasonal).

    Returns:
        MASE as a float. Values < 1 indicate better-than-naive performance.
        Returns NaN when y_train is too short to compute the naive scale or
        when the naive scale is zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= seasonality:
        warnings.warn(
            f"y_train length ({len(y_train)}) must be greater than seasonality "
            f"({seasonality}) to compute MASE. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0:
        return float("nan")

    return float(np.mean(np.abs(y_true - y_pred)) / scale)
