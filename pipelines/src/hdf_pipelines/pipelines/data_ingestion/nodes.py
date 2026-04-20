"""Data ingestion nodes: load raw CSV sources and apply basic cleaning."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_and_clean_demand(raw_demand: pd.DataFrame) -> pd.DataFrame:
    """Load and clean the raw demand time series.

    Args:
        raw_demand: Raw demand CSV loaded by the Kedro catalog. Expected columns
            include a date column and a numeric demand target for the critical SKU.

    Returns:
        Cleaned DataFrame with validated dtypes, sorted by date, and duplicates
        removed. Ready for downstream aggregation steps.
    """
    raise NotImplementedError(
        "Parse the date column to datetime, cast the target column to float, "
        "drop duplicate rows, sort by date ascending, and return the cleaned DataFrame."
    )


def load_and_clean_exogenous(raw_exogenous: pd.DataFrame) -> pd.DataFrame:
    """Load and clean exogenous variable time series.

    Args:
        raw_exogenous: Raw exogenous CSV loaded by the Kedro catalog. May include
            macroeconomic indicators, promotional flags, or weather signals.

    Returns:
        Cleaned DataFrame aligned by date with validated dtypes, no duplicate dates,
        and missing values handled via forward-fill or interpolation.
    """
    raise NotImplementedError(
        "Parse the date column, handle missing values via forward-fill, drop duplicates, "
        "sort by date, and return the cleaned exogenous DataFrame."
    )
