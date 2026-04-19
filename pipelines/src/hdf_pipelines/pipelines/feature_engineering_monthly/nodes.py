"""Feature engineering nodes for the monthly granularity."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_to_monthly(intermediate_demand: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned daily/weekly demand to calendar-month resolution.

    Args:
        intermediate_demand: Cleaned demand DataFrame from the ingestion layer.
            Must have a datetime index or a 'ds' date column and a numeric target.

    Returns:
        Monthly-aggregated demand DataFrame with one row per calendar month,
        summed target values, and a period-start date column.
    """
    raise NotImplementedError(
        "Resample the demand DataFrame to month-start frequency using pd.Grouper or "
        "resample('MS'), aggregate the target column with sum, and return the result."
    )


def build_monthly_features(
    monthly_demand: pd.DataFrame,
    exogenous_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Build lag, rolling, and calendar features for the monthly model input.

    Args:
        monthly_demand: Aggregated monthly demand from aggregate_to_monthly.
        exogenous_data: Cleaned exogenous DataFrame aligned to monthly frequency.
        parameters: Feature engineering config under the 'feature_engineering.monthly'
            key. Expected keys: lags, rolling_windows, calendar_features,
            include_exogenous.

    Returns:
        Feature-engineered DataFrame ready for model_input_preparation. Includes
        lag features, rolling statistics, calendar indicators (month, quarter),
        and optionally merged exogenous variables.
    """
    raise NotImplementedError(
        "Create lag features for each lag in parameters['lags'], rolling mean/std "
        "for each window in parameters['rolling_windows'], add month/quarter dummies "
        "if calendar_features=True, merge exogenous columns if include_exogenous=True, "
        "drop NaN rows introduced by lags, and return the feature DataFrame."
    )
