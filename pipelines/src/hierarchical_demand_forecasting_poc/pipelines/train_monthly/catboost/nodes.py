"""Monthly CatBoost training nodes: hyperparameter tuning and final model fit."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict,
) -> dict:
    """Run hyperparameter tuning for CatBoost on monthly data.

    Args:
        train: Feature-engineered monthly training set.
        validation: Held-out monthly validation set for candidate scoring.
        parameters: Content of train_monthly.yml under the 'train_monthly.catboost'
            key. Expected keys: 'enabled', 'tuning' (with iterations, learning_rate,
            depth grids).

    Returns:
        A dict with:
            - 'best_params': dict of chosen hyperparameters
            - 'cv_results': list of trial results with params and validation MAPE
            - 'best_score': float, best validation MAPE
    """
    raise NotImplementedError(
        "Iterate over parameters['tuning'] grids (iterations × learning_rate × depth), "
        "train CatBoostRegressor on 'train', score on 'validation' using MAPE from "
        "shared.metrics, and return the configuration with the lowest validation MAPE."
    )


def train_best_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tuning_result: dict,
) -> "CatBoostRegressor":
    """Refit CatBoost on train+validation using the best tuned hyperparameters.

    Args:
        train: Monthly training set.
        validation: Monthly validation set (included in final fit).
        tuning_result: Output of tune_hyperparameters containing 'best_params'.

    Returns:
        Fitted CatBoostRegressor ready to be evaluated on the held-out test set.
    """
    raise NotImplementedError(
        "Concatenate train and validation, instantiate CatBoostRegressor with "
        "tuning_result['best_params'], call model.fit(), and return the fitted model."
    )
