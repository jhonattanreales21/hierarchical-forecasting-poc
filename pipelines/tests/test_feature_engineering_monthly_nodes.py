"""Unit tests for monthly feature engineering pipeline nodes."""

import pandas as pd
import pytest

from hdf_pipelines.pipelines.feature_engineering_monthly.nodes import (
    build_monthly_calendar_features,
    build_monthly_exogenous_features,
    build_monthly_prophet_features,
)

_CALENDAR_FEATURE_COLUMNS = [
    "business_days",
    "total_tuesdays",
    "total_thursdays",
    "working_tuesdays",
    "working_thursdays",
    "has_5_working_tuesdays",
    "has_5_working_thursdays",
    "tuesday_holidays",
    "thursday_holidays",
    "total_holidays",
]


def _feature_engineering_parameters() -> dict:
    """Return a minimal feature engineering parameter config for testing."""
    return {
        "date_column": "month_start_date",
        "sku_column": "sku",
        "target_column": "monthly_demand",
        "calendar_features": {
            "enabled": True,
            "country_holidays": "CO",
            "observed_holidays": True,
            "weekmask": "Mon Tue Wed Thu Fri",
        },
        "exogenous_features": {
            "enabled": True,
            "base_columns": ["pfizer_limited"],
            "lags": [1],
        },
    }


def _demand_monthly_df(periods: int = 3) -> pd.DataFrame:
    """Return a minimal monthly demand DataFrame for the given number of periods."""
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=periods, freq="MS"),
            "sku": ["SKU-1"] * periods,
            "monthly_demand": [float(100 + i * 10) for i in range(periods)],
        }
    )


def _exogenous_monthly_df(periods: int = 3) -> pd.DataFrame:
    """Return a minimal monthly exogenous DataFrame for the given number of periods."""
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=periods, freq="MS"),
            "pfizer_limited": ([0.0, 1.0, 0.0] * periods)[:periods],
        }
    )


def _calendar_df(periods: int = 3) -> pd.DataFrame:
    """Return a manually-constructed calendar feature DataFrame to avoid cross-test coupling."""
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=periods, freq="MS"),
            "business_days": ([23, 21, 21] * periods)[:periods],
            "total_tuesdays": ([5, 4, 5] * periods)[:periods],
            "total_thursdays": ([4, 4, 4] * periods)[:periods],
            "working_tuesdays": ([5, 4, 5] * periods)[:periods],
            "working_thursdays": ([4, 4, 4] * periods)[:periods],
            "has_5_working_tuesdays": ([1, 0, 1] * periods)[:periods],
            "has_5_working_thursdays": ([0, 0, 0] * periods)[:periods],
            "tuesday_holidays": ([0, 0, 0] * periods)[:periods],
            "thursday_holidays": ([0, 0, 0] * periods)[:periods],
            "total_holidays": ([1, 0, 1] * periods)[:periods],
        }
    )


def test_build_monthly_calendar_features_raises_when_disabled():
    """Passing enabled=False raises ValueError before any computation is attempted."""
    params = _feature_engineering_parameters()
    params["calendar_features"]["enabled"] = False

    with pytest.raises(
        ValueError,
        match="feature_engineering_monthly.calendar_features.enabled must be true",
    ):
        build_monthly_calendar_features(_demand_monthly_df(), params)


def test_build_monthly_calendar_features_returns_one_row_per_unique_month():
    """Output is deduplicated to one row per month regardless of how many SKUs are in demand."""
    # 2 SKUs × 4 months = 8 input rows; calendar output must collapse to 4 unique months
    demand_df = pd.DataFrame(
        {
            "month_start_date": pd.date_range(
                "2024-01-01", periods=4, freq="MS"
            ).tolist()
            * 2,
            "sku": ["SKU-1"] * 4 + ["SKU-2"] * 4,
            "monthly_demand": [100.0] * 8,
        }
    )

    result = build_monthly_calendar_features(
        demand_df, _feature_engineering_parameters()
    )

    assert len(result) == 4
    assert "month_start_date" in result.columns
    for col in _CALENDAR_FEATURE_COLUMNS:
        assert col in result.columns
    assert result["business_days"].dtype == "int64"


def test_build_monthly_exogenous_features_appends_lag_columns_with_leading_nulls():
    """Lag columns are created and the first row carries NaN because there is no prior month."""
    # pfizer_limited = [0.0, 1.0, 0.0] → lag_1 = [NaN, 0.0, 1.0]
    result = build_monthly_exogenous_features(
        _exogenous_monthly_df(periods=3),
        _feature_engineering_parameters(),
    )

    assert "pfizer_limited_lag_1" in result.columns
    assert pd.isna(result["pfizer_limited_lag_1"].iloc[0])
    assert result["pfizer_limited_lag_1"].iloc[1] == 0.0  # carries Jan value into Feb


def test_build_monthly_prophet_features_merges_demand_calendar_and_exogenous():
    """Left-joining all three sources produces one row per (SKU, month) with all feature columns."""
    demand_df = _demand_monthly_df(periods=3)

    result = build_monthly_prophet_features(
        demand_df,
        _calendar_df(periods=3),
        _exogenous_monthly_df(periods=3),
        _feature_engineering_parameters(),
    )

    assert len(result) == 3
    assert "monthly_demand" in result.columns
    assert "business_days" in result.columns
    assert "pfizer_limited" in result.columns


def test_build_monthly_prophet_features_raises_on_duplicate_calendar_month_keys():
    """Duplicate month_start_date rows in the calendar table raise ValueError before the join."""
    demand_df = _demand_monthly_df(periods=2)
    calendar_with_duplicate = pd.concat(
        [_calendar_df(periods=2), _calendar_df(periods=1)],  # first month appears twice
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="must be unique by"):
        build_monthly_prophet_features(
            demand_df,
            calendar_with_duplicate,
            _exogenous_monthly_df(periods=2),
            _feature_engineering_parameters(),
        )
