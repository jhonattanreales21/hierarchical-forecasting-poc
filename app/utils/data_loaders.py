"""Centralised data loading helpers for the Streamlit app.

All loaders return empty DataFrames or empty dicts for missing artifacts rather
than raising errors. Pages decide how to display missing-artifact warnings.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.champion import (
    standardize_champion_metadata,
    standardize_forecast_columns,
)
from utils.descriptive import normalize_demand_frame, normalize_exogenous_frame
from utils.paths import (
    ACTUALS,
    CANDIDATE_METRICS,
    CHAMPION_META,
    DEMAND_DAILY,
    DEMAND_MONTHLY,
    DEMAND_WEEKLY,
    EXOGENOUS_MONTHLY,
    EXPLAINABILITY_META,
    FAMILY_CHAMPION_IMPORTANCE,
    FAMILY_CHAMPION_SUMMARY,
    INFERENCE_META,
    RAW_EXOGENOUS,
    SELECTION_SUMMARY,
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

    Supports both the family-agnostic schema (month_start_date / monthly_demand)
    and the legacy Prophet schema (ds / y).

    Returns:
        DataFrame with ds (datetime) and y columns, sorted by ds.
    """
    if not ACTUALS.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ACTUALS)
    if "ds" not in df.columns and "month_start_date" in df.columns:
        df = df.rename(columns={"month_start_date": "ds"})
    if "y" not in df.columns and "monthly_demand" in df.columns:
        df = df.rename(columns={"monthly_demand": "y"})
    if "ds" not in df.columns or "y" not in df.columns:
        return pd.DataFrame()
    df = df[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)


@st.cache_data
def load_monthly_modeling_data_full() -> pd.DataFrame:
    """Load monthly modeling data with all available feature columns."""
    if not ACTUALS.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ACTUALS)
    if "ds" not in df.columns and "month_start_date" in df.columns:
        df = df.rename(columns={"month_start_date": "ds"})
    if "y" not in df.columns and "monthly_demand" in df.columns:
        df = df.rename(columns={"monthly_demand": "y"})
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")
    return df.reset_index(drop=True)


@st.cache_data
def load_raw_exogenous_data() -> pd.DataFrame:
    """Load raw monthly exogenous variables if available."""
    if not RAW_EXOGENOUS.exists():
        return pd.DataFrame()
    df = pd.read_csv(RAW_EXOGENOUS)
    df.columns = [str(col).strip() for col in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.reset_index(drop=True)


@st.cache_data
def load_demand_daily() -> pd.DataFrame:
    """Load canonical daily demand for descriptive analysis."""
    if not DEMAND_DAILY.exists():
        return pd.DataFrame(columns=["date", "sku", "demand", "granularity"])
    return normalize_demand_frame(pd.read_parquet(DEMAND_DAILY), "daily")


@st.cache_data
def load_demand_weekly() -> pd.DataFrame:
    """Load canonical weekly demand for descriptive analysis."""
    if not DEMAND_WEEKLY.exists():
        return pd.DataFrame(columns=["date", "sku", "demand", "granularity"])
    return normalize_demand_frame(pd.read_parquet(DEMAND_WEEKLY), "weekly")


@st.cache_data
def load_demand_monthly_primary() -> pd.DataFrame:
    """Load canonical monthly demand for descriptive analysis."""
    if not DEMAND_MONTHLY.exists():
        return pd.DataFrame(columns=["date", "sku", "demand", "granularity"])
    return normalize_demand_frame(pd.read_parquet(DEMAND_MONTHLY), "monthly")


@st.cache_data
def load_exogenous_monthly_primary() -> pd.DataFrame:
    """Load canonical monthly exogenous variables for descriptive analysis."""
    if not EXOGENOUS_MONTHLY.exists():
        return pd.DataFrame()
    return normalize_exogenous_frame(pd.read_parquet(EXOGENOUS_MONTHLY))


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
    df = standardize_forecast_columns(df)
    if "ds" not in df.columns:
        return df
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
def load_candidate_metrics() -> pd.DataFrame:
    """Load per-candidate rolling-origin metrics.

    Returns:
        DataFrame with evaluation metrics per candidate, or empty DataFrame if missing.
    """
    if not CANDIDATE_METRICS.exists():
        return pd.DataFrame()
    return pd.read_parquet(CANDIDATE_METRICS)


@st.cache_data
def load_family_champion_summary() -> pd.DataFrame:
    """Load the per-family champion summary (best candidate per model family).

    Returns:
        DataFrame with one row per family, or empty DataFrame if missing.
    """
    if not FAMILY_CHAMPION_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_parquet(FAMILY_CHAMPION_SUMMARY)


@st.cache_data
def load_family_champion_importance() -> pd.DataFrame:
    """Load the unified family-champion driver-importance table.

    One row per (family, feature) with ``importance``, ``importance_type``, and ``rank``.
    SHAP mean(|value|) for CatBoost; native contribution/coefficient drivers for the
    other families.

    Returns:
        Long-form DataFrame, or empty DataFrame if the artifact is missing.
    """
    if not FAMILY_CHAMPION_IMPORTANCE.exists():
        return pd.DataFrame()
    return pd.read_parquet(FAMILY_CHAMPION_IMPORTANCE)


def load_champion_metadata() -> dict:
    """Load champion model metadata JSON.

    Returns:
        Dict with champion metadata, or empty dict if missing.
    """
    return standardize_champion_metadata(load_json(CHAMPION_META))


def load_explainability_metadata() -> dict:
    """Load family-champion explainability metadata JSON.

    Returns:
        Dict with per-family explainability method/provenance, or empty dict if missing.
    """
    return load_json(EXPLAINABILITY_META)


def load_inference_metadata() -> dict:
    """Load inference run metadata JSON.

    Returns:
        Dict with inference run details, or empty dict if missing.
    """
    return load_json(INFERENCE_META)
