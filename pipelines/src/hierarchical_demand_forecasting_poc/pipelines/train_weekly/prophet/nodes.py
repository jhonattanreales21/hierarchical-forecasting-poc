"""Weekly Prophet training nodes: hyperparameter tuning and final model fit."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict,
) -> dict:
    """Run hyperparameter tuning for Prophet on weekly data.

    Args:
        train: Weekly training set in Prophet format (ds, y, optional regressors).
        validation: Held-out weekly validation set for candidate scoring.
        parameters: Content of train_weekly.yml under the 'train_weekly.prophet'
            key. Expected keys: 'enabled', 'tuning' (with changepoint_prior_scale,
            seasonality_prior_scale, seasonality_mode grids).

    Returns:
        A dict with:
            - 'best_params': dict of chosen hyperparameters
            - 'cv_results': list of trial results with params and validation MAPE
            - 'best_score': float, best validation MAPE
    """
    raise NotImplementedError(
        "Iterate over the Cartesian product of parameters['tuning'] grids, "
        "instantiate Prophet with each combination, fit on 'train', predict on "
        "'validation' dates, compute MAPE using shared.metrics.mape, and return "
        "the configuration with the lowest validation MAPE."
    )


def train_best_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tuning_result: dict,
) -> "Prophet":
    """Refit Prophet on train+validation using the best tuned hyperparameters.

    Args:
        train: Weekly training set in Prophet format.
        validation: Weekly validation set (included in final fit).
        tuning_result: Output of tune_hyperparameters containing 'best_params'.

    Returns:
        Fitted Prophet model ready to be evaluated on the held-out test set.
    """
    raise NotImplementedError(
        "Concatenate train and validation DataFrames, instantiate Prophet with "
        "tuning_result['best_params'], call model.fit(), and return the fitted model."
    )
