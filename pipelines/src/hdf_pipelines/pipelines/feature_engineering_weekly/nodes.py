"""Feature engineering nodes for the weekly granularity."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_to_weekly(intermediate_demand: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned daily demand to ISO-week resolution.

    Args:
        intermediate_demand: Cleaned demand DataFrame from the ingestion layer.
            Must have a datetime index or a 'ds' date column and a numeric target.

    Returns:
        Weekly-aggregated demand DataFrame with one row per ISO week (Monday-start),
        summed target values, and a week-start date column.
    """
    raise NotImplementedError(
        "Resample the demand DataFrame to week-start (W-MON) frequency using "
        "resample('W-MON'), aggregate the target column with sum, and return the result."
    )


def build_weekly_features(
    weekly_demand: pd.DataFrame,
    exogenous_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Build lag, rolling, and calendar features for the weekly model input.

    Args:
        weekly_demand: Aggregated weekly demand from aggregate_to_weekly.
        exogenous_data: Cleaned exogenous DataFrame resampled to weekly frequency.
        parameters: Feature engineering config under the 'feature_engineering.weekly'
            key. Expected keys: lags, rolling_windows, calendar_features,
            include_exogenous.

    Returns:
        Feature-engineered DataFrame ready for model_input_preparation. Includes
        lag features (e.g., 1, 2, 4, 8, 52 weeks), rolling statistics, week-of-year
        and month indicators, and optionally merged exogenous variables.
    """
    raise NotImplementedError(
        "Create lag features for each lag in parameters['lags'], rolling mean/std "
        "for each window in parameters['rolling_windows'], add week_of_year/month "
        "dummies if calendar_features=True, merge exogenous columns if "
        "include_exogenous=True, drop NaN rows, and return the feature DataFrame."
    )
