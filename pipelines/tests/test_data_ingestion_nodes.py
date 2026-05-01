"""Unit tests for data ingestion pipeline nodes."""

import pandas as pd
import pytest

from hdf_pipelines.pipelines.data_ingestion.nodes import (
    build_demand_weekly,
    load_and_clean_demand,
    load_and_clean_exogenous,
    mask_raw_demand,
)
from hdf_pipelines.pipelines.data_ingestion.pipeline import create_pipeline

EXPECTED_EXOGENOUS_ROWS = 2
EXPECTED_WEEKLY_ROWS = 2
EXPECTED_FIRST_WEEK_TOTAL = 15.0
EXPECTED_SECOND_WEEK_TOTAL = 8.0


def _raw_demand_df() -> pd.DataFrame:
    """Return a minimal valid raw demand DataFrame matching the source CSV column contract."""
    return pd.DataFrame(
        {
            "SKU": ["SKU-1", "SKU-1", "SKU-1"],
            "Year": [2024, 2024, 2024],
            "Month": ["2024-01", "2024-01", "2024-02"],
            "Month Name": ["January", "January", "February"],
            "Date": ["2024-01-08", "2024-01-09", "2024-02-05"],
            "Monthly Demand": [100.0, 100.0, 120.0],
            "Daily Demand": [5.0, 6.0, 4.0],
        }
    )


def test_load_and_clean_demand_raises_on_missing_columns():
    """A raw demand DataFrame missing any expected column raises ValueError."""
    df = _raw_demand_df().drop(columns=["Daily Demand"])

    with pytest.raises(ValueError, match="Missing required columns in raw demand data"):
        load_and_clean_demand(df)


def test_mask_raw_demand_anonymizes_skus_and_scales_demand_columns():
    """Raw demand is anonymized before cleaning and both demand measures are scaled."""
    df = pd.DataFrame(
        {
            "SKU": [" Real SKU A ", "Real SKU A", "Real SKU B"],
            "Year": [2024, 2024, 2024],
            "Month": ["2024-01", "2024-01", "2024-02"],
            "Month Name": ["January", "January", "February"],
            "Date": ["2024-01-08", "2024-01-09", "2024-02-05"],
            "Monthly Demand": [1000.0, 1000.0, 2500.0],
            "Daily Demand": [50.0, 75.0, 25.0],
        }
    )

    result = mask_raw_demand(df)

    assert result["SKU"].tolist() == ["sku1", "sku1", "sku2"]
    assert result["Monthly Demand"].tolist() == [10.0, 10.0, 25.0]
    assert result["Daily Demand"].tolist() == [0.5, 0.75, 0.25]


def test_load_and_clean_exogenous_strips_trailing_whitespace_from_column_names():
    """Column names with trailing spaces are normalised before the contract check.

    The real source CSV ships with "surgifoam_limited " (trailing space). Stripping
    must happen before column validation, otherwise a valid file would be rejected.
    """
    df = pd.DataFrame(
        {
            "Date": ["2024-01", "2024-02"],
            "pfizer_limited": [0.0, 1.0],
            "surgifoam_limited ": [0.0, 0.0],  # intentional trailing space
            "rebate_target": [0.5, 0.5],
        }
    )

    result = load_and_clean_exogenous(df)

    assert "surgifoam_limited" in result.columns
    assert "month_start_date" in result.columns
    assert len(result) == EXPECTED_EXOGENOUS_ROWS


def test_build_demand_weekly_snaps_dates_to_monday_and_sums_daily_demand():
    """Daily rows within the same ISO week are aggregated to a Monday-anchored weekly total."""
    # 2024-01-08 = Monday, 2024-01-09 = Tuesday (same week); 2024-01-15 = next Monday
    demand_cleaned = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-15"]),
            "sku": ["SKU-1", "SKU-1", "SKU-1"],
            "daily_demand": [10.0, 5.0, 8.0],
            "monthly_demand": [100.0, 100.0, 100.0],
            "year": [2024, 2024, 2024],
            "month": [1, 1, 1],
            "month_name": ["January", "January", "January"],
        }
    )

    result = build_demand_weekly(demand_cleaned)

    assert list(result.columns) == ["week_start_date", "sku", "weekly_demand"]
    assert len(result) == EXPECTED_WEEKLY_ROWS  # two distinct ISO weeks
    week1 = result[result["week_start_date"] == pd.Timestamp("2024-01-08")]
    assert week1["weekly_demand"].iloc[0] == EXPECTED_FIRST_WEEK_TOTAL  # 10 + 5
    week2 = result[result["week_start_date"] == pd.Timestamp("2024-01-15")]
    assert week2["weekly_demand"].iloc[0] == EXPECTED_SECOND_WEEK_TOTAL


def test_data_ingestion_pipeline_starts_with_masking_node():
    """The ingestion pipeline should mask raw demand before cleaning it."""
    ingestion = create_pipeline()
    node_names = [node.name for node in ingestion.nodes]

    assert "mask_raw_demand" in node_names
    assert "load_and_clean_demand" in node_names
    assert node_names.index("mask_raw_demand") < node_names.index(
        "load_and_clean_demand"
    )
