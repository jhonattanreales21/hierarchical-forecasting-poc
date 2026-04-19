"""Weekly SARIMAX training nodes: hyperparameter tuning and final model fit."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict,
) -> dict:
    """Run order-grid search for SARIMAX on weekly data.

    Args:
        train: Weekly training set with target column and exogenous variables.
        validation: Held-out weekly validation set for candidate scoring.
        parameters: Content of train_weekly.yml under the 'train_weekly.sarimax'
            key. Expected keys: 'enabled', 'tuning' (with order_grid and
            seasonal_order_grid, where s=52 for weekly annual seasonality).

    Returns:
        A dict with:
            - 'best_order': list [p, d, q]
            - 'best_seasonal_order': list [P, D, Q, s]
            - 'cv_results': list of trial results
            - 'best_score': float, best validation MAPE
    """
    raise NotImplementedError(
        "Iterate over order_grid × seasonal_order_grid, fit SARIMAX on 'train', "
        "evaluate one-step-ahead forecasts on 'validation', compute MAPE, and return "
        "the configuration with the lowest validation MAPE. Note s=52 for weekly data."
    )


def train_best_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tuning_result: dict,
) -> "SARIMAXResultsWrapper":
    """Refit SARIMAX on train+validation using the best tuned order parameters.

    Args:
        train: Weekly training set.
        validation: Weekly validation set (included in final fit).
        tuning_result: Output of tune_hyperparameters.

    Returns:
        Fitted SARIMAXResultsWrapper ready to be evaluated on the held-out test set.
    """
    raise NotImplementedError(
        "Concatenate train and validation, instantiate SARIMAX with "
        "tuning_result['best_order'] and tuning_result['best_seasonal_order'], "
        "call results = model.fit(disp=False), and return results."
    )
