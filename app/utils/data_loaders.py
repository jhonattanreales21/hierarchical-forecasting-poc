"""Centralised data loading helpers for the Streamlit app.

All loaders return empty DataFrames or empty dicts for missing artifacts rather
than raising errors. Pages decide how to display missing-artifact warnings.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.paths import (
    ACTUALS,
    CHAMPION_META,
    INFERENCE_META,
    SELECTION_FORECAST,
    SELECTION_SUMMARY,
    TEST_METRICS,
    forecast_parquet,
)


@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file, returning an empty DataFrame if the file is missing.

    Args:
        path: Absolute path to the parquet file.

    Returns:
        DataFrame with file contents, or empty DataFrame if missing.
    """
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_json(path: Path) -> dict:
    """Load a JSON file, returning an empty dict if the file is missing.

    Args:
        path: Absolute path to the JSON file.

    Returns:
        Parsed dict, or empty dict if missing.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_monthly_modeling_data() -> pd.DataFrame:
    """Load monthly modeling data (actuals), returning only ds and y columns.

    Returns:
        DataFrame with ds (datetime) and y columns, sorted by ds.
    """
    if not ACTUALS.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ACTUALS, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)


@st.cache_data
def load_champion_test_forecast() -> pd.DataFrame:
    """Load the full champion test forecast with ds parsed as datetime.

    Returns:
        DataFrame with all candidates, sorted by ds.
    """
    if not SELECTION_FORECAST.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SELECTION_FORECAST)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)


@st.cache_data
def load_monthly_forecast(horizon_months: int) -> pd.DataFrame:
    """Load a horizon-specific future forecast with ds parsed as datetime.

    Args:
        horizon_months: Forecast horizon in months (e.g. 3 or 6).

    Returns:
        DataFrame sorted by ds, or empty DataFrame if the artifact is missing.
    """
    path = forecast_parquet(horizon_months)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)


@st.cache_data
def load_model_selection_summary() -> pd.DataFrame:
    """Load the model selection comparison summary.

    Returns:
        DataFrame with one row per candidate, or empty DataFrame if missing.
    """
    if not SELECTION_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_parquet(SELECTION_SUMMARY)


@st.cache_data
def load_test_metrics() -> pd.DataFrame:
    """Load per-candidate test metrics.

    Returns:
        DataFrame with evaluation metrics per candidate, or empty DataFrame if missing.
    """
    if not TEST_METRICS.exists():
        return pd.DataFrame()
    return pd.read_parquet(TEST_METRICS)


def load_champion_metadata() -> dict:
    """Load champion model metadata JSON.

    Returns:
        Dict with champion metadata, or empty dict if missing.
    """
    return load_json(CHAMPION_META)


def load_inference_metadata() -> dict:
    """Load inference run metadata JSON.

    Returns:
        Dict with inference run details, or empty dict if missing.
    """
    return load_json(INFERENCE_META)
