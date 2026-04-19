"""Shared utilities for the weekly training pipeline."""

import logging

import numpy as np

from shared.metrics import mape

logger = logging.getLogger(__name__)


def compute_validation_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE between actual and predicted values on a validation fold.

    Args:
        y_true: Array of actual observed values.
        y_pred: Array of model predictions, same shape as y_true.

    Returns:
        MAPE as a float. Lower is better.
    """
    return mape(y_true, y_pred)
