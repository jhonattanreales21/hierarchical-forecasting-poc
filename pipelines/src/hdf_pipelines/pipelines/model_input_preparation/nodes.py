"""Model input preparation nodes: time-aware train/val/test splits and backtest folds."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_monthly_splits_prophet(
    feature_monthly: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create Prophet-specific train/validation/test splits for monthly data.

    Args:
        feature_monthly: Feature-engineered monthly DataFrame from feature engineering.
        parameters: Config from model_input key. Expected keys: validation_size_months,
            test_size_months.

    Returns:
        Tuple of (train, validation, test) DataFrames in chronological order.
        Prophet format requires 'ds' (date) and 'y' (target) columns plus regressors.
    """
    raise NotImplementedError(
        "Sort by date, slice the last test_size_months rows as test, the preceding "
        "validation_size_months rows as validation, and the remainder as train. "
        "Rename columns to Prophet format (ds, y) and return the three splits."
    )


def build_monthly_splits_catboost(
    feature_monthly: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create CatBoost-specific train/validation/test splits for monthly data.

    Args:
        feature_monthly: Feature-engineered monthly DataFrame.
        parameters: Config from model_input key.

    Returns:
        Tuple of (train, validation, test) DataFrames with all lag/rolling features
        retained as columns. Target column kept as-is for CatBoost.
    """
    raise NotImplementedError(
        "Apply the same chronological split logic as build_monthly_splits_prophet "
        "but keep all feature columns without renaming. Return (train, validation, test)."
    )


def build_monthly_splits_sarimax(
    feature_monthly: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create SARIMAX-specific train/validation/test splits for monthly data.

    Args:
        feature_monthly: Feature-engineered monthly DataFrame.
        parameters: Config from model_input key.

    Returns:
        Tuple of (train, validation, test) DataFrames. The exogenous columns are kept
        separate for SARIMAX's exog argument during fitting.
    """
    raise NotImplementedError(
        "Apply chronological splits and return (train, validation, test). "
        "Ensure the target series and exogenous matrix are consistently ordered."
    )


def build_weekly_splits_prophet(
    feature_weekly: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create Prophet-specific train/validation/test splits for weekly data.

    Args:
        feature_weekly: Feature-engineered weekly DataFrame.
        parameters: Config from model_input key. Expected keys: validation_size_weeks,
            test_size_weeks.

    Returns:
        Tuple of (train, validation, test) DataFrames in Prophet format (ds, y).
    """
    raise NotImplementedError(
        "Sort by date, slice the last test_size_weeks rows as test, the preceding "
        "validation_size_weeks rows as validation, and the remainder as train. "
        "Rename to Prophet format and return the three splits."
    )


def build_weekly_splits_catboost(
    feature_weekly: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create CatBoost-specific train/validation/test splits for weekly data.

    Args:
        feature_weekly: Feature-engineered weekly DataFrame.
        parameters: Config from model_input key.

    Returns:
        Tuple of (train, validation, test) DataFrames with all feature columns retained.
    """
    raise NotImplementedError(
        "Apply chronological weekly splits keeping all feature columns. "
        "Return (train, validation, test)."
    )


def build_weekly_splits_sarimax(
    feature_weekly: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create SARIMAX-specific train/validation/test splits for weekly data.

    Args:
        feature_weekly: Feature-engineered weekly DataFrame.
        parameters: Config from model_input key.

    Returns:
        Tuple of (train, validation, test) DataFrames with exogenous columns intact.
    """
    raise NotImplementedError(
        "Apply chronological weekly splits and return (train, validation, test). "
        "Preserve column order for consistent exog matrix construction."
    )


def generate_backtest_folds_monthly(
    feature_monthly: pd.DataFrame,
    parameters: dict,
) -> list[dict[str, Any]]:
    """Generate rolling-origin backtest fold definitions for monthly data.

    Args:
        feature_monthly: Feature-engineered monthly DataFrame.
        parameters: Config from model_input key. Expected keys: n_folds,
            backtesting_strategy, validation_size_months, test_size_months.

    Returns:
        List of fold dicts, each with 'train_end', 'val_end', 'test_end' date strings
        defining the boundaries of each rolling-origin fold.
    """
    raise NotImplementedError(
        "Implement rolling-origin splits: for each fold k in range(n_folds), "
        "shift the origin backward by validation_size_months * k rows. "
        "Return a list of dicts with date boundary keys for each fold."
    )


def generate_backtest_folds_weekly(
    feature_weekly: pd.DataFrame,
    parameters: dict,
) -> list[dict[str, Any]]:
    """Generate rolling-origin backtest fold definitions for weekly data.

    Args:
        feature_weekly: Feature-engineered weekly DataFrame.
        parameters: Config from model_input key. Expected keys: n_folds,
            backtesting_strategy, validation_size_weeks, test_size_weeks.

    Returns:
        List of fold dicts with 'train_end', 'val_end', 'test_end' date strings.
    """
    raise NotImplementedError(
        "Implement rolling-origin splits for weekly data using test_size_weeks "
        "and validation_size_weeks as offsets. Return a list of fold boundary dicts."
    )
