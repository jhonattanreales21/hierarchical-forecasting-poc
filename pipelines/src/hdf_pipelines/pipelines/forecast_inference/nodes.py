"""Forecast inference nodes: generate forward-looking predictions from champion models."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_inference_inputs(
    feature_monthly: pd.DataFrame,
    feature_weekly: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the input DataFrames for inference (no target leakage).

    Args:
        feature_monthly: Full monthly feature DataFrame including the most recent rows.
        feature_weekly: Full weekly feature DataFrame including the most recent rows.
        parameters: Config from forecast_inference key. Expected keys: mode,
            include_intervals, confidence_level.

    Returns:
        Tuple of (monthly_inference_df, weekly_inference_df) containing only the
        feature columns required for prediction, with future date rows appended
        for the forecast horizon.
    """
    raise NotImplementedError(
        "Identify the most recent date in each granularity, extend the date range "
        "by the configured horizon, build future feature rows (lag features from "
        "the most recent actuals), and return the inference-ready DataFrames."
    )


def generate_monthly_forecast(
    champion_monthly_model: Any,
    monthly_inference_df: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Generate monthly demand forecasts using the elected champion model.

    Args:
        champion_monthly_model: Serialised champion model from data/06_models/champions/.
        monthly_inference_df: Inference-ready monthly DataFrame from load_inference_inputs.
        parameters: Config from forecast_inference key with include_intervals and
            confidence_level.

    Returns:
        DataFrame with columns [ds, y_forecast, y_lower, y_upper, model_name,
        granularity, horizon_step] for each forecast month.
    """
    raise NotImplementedError(
        "Dispatch to the appropriate prediction method based on model type (Prophet: "
        "model.predict(); CatBoost: model.predict(); SARIMAX: results.forecast()). "
        "If include_intervals=True, compute prediction intervals at confidence_level. "
        "Return a standardised forecast DataFrame."
    )


def generate_weekly_forecast(
    champion_weekly_model: Any,
    weekly_inference_df: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Generate weekly demand forecasts using the elected champion model.

    Args:
        champion_weekly_model: Serialised champion model from data/06_models/champions/.
        weekly_inference_df: Inference-ready weekly DataFrame from load_inference_inputs.
        parameters: Config from forecast_inference key with include_intervals and
            confidence_level.

    Returns:
        DataFrame with columns [ds, y_forecast, y_lower, y_upper, model_name,
        granularity, horizon_step] for each forecast week.
    """
    raise NotImplementedError(
        "Apply the same dispatch logic as generate_monthly_forecast but for the weekly "
        "champion model. Return a standardised forecast DataFrame with granularity='weekly'."
    )


def allocate_daily_forecast(
    forecast_weekly_reconciled: pd.DataFrame,
    feature_weekly: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Disaggregate reconciled weekly forecasts to daily estimates.

    Args:
        forecast_weekly_reconciled: Reconciled weekly forecast DataFrame.
        feature_weekly: Weekly feature DataFrame used to compute historical daily shares.
        parameters: Config from forecast_inference key. Expected key: daily_allocation
            with 'enabled' (bool) and 'method' ('historical_shares').

    Returns:
        DataFrame with daily forecast rows, or an empty DataFrame if
        daily_allocation.enabled is False.
    """
    raise NotImplementedError(
        "If parameters['daily_allocation']['enabled'] is False, return an empty "
        "DataFrame. Otherwise, compute historical day-of-week share fractions from "
        "feature_weekly and distribute each week's forecast proportionally across "
        "7 daily rows."
    )
