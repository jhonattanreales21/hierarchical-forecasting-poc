"""Monthly SARIMAX training nodes: hyperparameter tuning and final model fit."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict,
) -> dict:
    """Run order-grid search for SARIMAX on monthly data.

    Args:
        train: Monthly training set with target column and exogenous variables.
        validation: Held-out monthly validation set for candidate scoring.
        parameters: Content of train_monthly.yml under the 'train_monthly.sarimax'
            key. Expected keys: 'enabled', 'tuning' (with order_grid and
            seasonal_order_grid as lists of [p, d, q] and [P, D, Q, s]).

    Returns:
        A dict with:
            - 'best_order': list [p, d, q]
            - 'best_seasonal_order': list [P, D, Q, s]
            - 'cv_results': list of trial results with AIC/BIC and validation MAPE
            - 'best_score': float, best validation MAPE
    """
    raise NotImplementedError(
        "Iterate over the Cartesian product of order_grid × seasonal_order_grid, "
        "fit SARIMAX on 'train' with each combination using statsmodels, evaluate "
        "one-step-ahead forecasts on 'validation', compute MAPE, and return the "
        "configuration with the lowest validation MAPE."
    )


def train_best_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tuning_result: dict,
) -> "SARIMAXResultsWrapper":
    """Refit SARIMAX on train+validation using the best tuned order parameters.

    Args:
        train: Monthly training set.
        validation: Monthly validation set (included in final fit).
        tuning_result: Output of tune_hyperparameters with 'best_order' and
            'best_seasonal_order'.

    Returns:
        Fitted SARIMAXResultsWrapper ready to be evaluated on the held-out test set.
    """
    raise NotImplementedError(
        "Concatenate train and validation, instantiate SARIMAX with "
        "tuning_result['best_order'] and tuning_result['best_seasonal_order'], "
        "call results = model.fit(disp=False), and return results."
    )
