"""Weekly CatBoost training nodes: hyperparameter tuning and final model fit."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict,
) -> dict:
    """Run hyperparameter tuning for CatBoost on weekly data.

    Args:
        train: Feature-engineered weekly training set.
        validation: Held-out validation set for candidate scoring.
        parameters: Content of train_weekly.yml under the 'train_weekly.catboost'
            key. Expected keys: 'enabled', 'tuning' (with iterations, learning_rate,
            depth grids).

    Returns:
        A dict with:
            - 'best_params': dict of chosen hyperparameters
            - 'cv_results': list of trial results
            - 'best_score': float, best validation metric
    """
    raise NotImplementedError(
        "Iterate over parameters['tuning'] grids, train CatBoost on `train`, "
        "score on `validation` using MAPE, and return best configuration."
    )


def train_best_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tuning_result: dict,
) -> "CatBoostRegressor":
    """Refit CatBoost on train+validation using the best tuned params.

    Args:
        train: Training set.
        validation: Validation set (included in final fit).
        tuning_result: Output of tune_hyperparameters.

    Returns:
        Trained CatBoostRegressor ready to be evaluated on test data.
    """
    raise NotImplementedError(
        "Concatenate train and validation, instantiate CatBoostRegressor with "
        "tuning_result['best_params'], fit, and return the model."
    )
