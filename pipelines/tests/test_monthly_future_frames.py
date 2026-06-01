"""Tests for generic monthly future frame generation.

Covers:
- month_start_date is the canonical date column (not ds)
- target column is absent from future frames
- each horizon frame contains exactly the right number of future months
- future frames start strictly after the last historical month
- all configured active regressors are present
- sku column is present
- unknown active regressors raise a clear error
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hdf_pipelines.pipelines.model_input_preparation.nodes import (
    build_monthly_generic_future_frames,
)

_CALENDAR_COLUMNS = [
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


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_modeling_data(n: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "month_start_date": dates,
            "monthly_demand": np.linspace(100.0, 200.0, n),
            "sku": ["SKU_001"] * n,
            "business_days": [22] * n,
        }
    )


def _make_calendar_features(n: int = 40) -> pd.DataFrame:
    """Pre-populate calendar features far enough to cover all future horizons."""
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "month_start_date": dates,
            "business_days": [22] * n,
            "total_tuesdays": [4] * n,
            "total_thursdays": [4] * n,
            "working_tuesdays": [4] * n,
            "working_thursdays": [4] * n,
            "has_5_working_tuesdays": [0] * n,
            "has_5_working_thursdays": [0] * n,
            "tuesday_holidays": [0] * n,
            "thursday_holidays": [0] * n,
            "total_holidays": [0] * n,
        }
    )


def _make_exogenous_features() -> pd.DataFrame:
    """Minimal exogenous table — no exogenous regressors active in these tests."""
    return pd.DataFrame(
        {"month_start_date": pd.date_range("2023-01-01", periods=1, freq="MS")}
    )


def _make_params(active_regressors: list[str] | None = None) -> dict:
    return {
        "monthly": {
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "sku_column": "sku",
            "active_regressors": active_regressors or ["business_days"],
        }
    }


def _make_calendar_params() -> dict:
    return {
        "calendar_features": {
            "country_holidays": "CO",
            "observed_holidays": True,
            "weekmask": "Mon Tue Wed Thu Fri",
        }
    }


def _build_frames(n_history: int = 24, active_regressors: list[str] | None = None):
    return build_monthly_generic_future_frames(
        monthly_modeling_data=_make_modeling_data(n_history),
        monthly_calendar_features=_make_calendar_features(40),
        monthly_exogenous_features=_make_exogenous_features(),
        parameters=_make_params(active_regressors),
        calendar_parameters=_make_calendar_params(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_generic_future_frames_use_month_start_date_not_ds():
    """Canonical date column is month_start_date, not the Prophet-specific ds."""
    f3, f6, f12 = _build_frames()
    for frame, label in ((f3, "3m"), (f6, "6m"), (f12, "12m")):
        assert "month_start_date" in frame.columns, (
            f"monthly_future_{label} is missing month_start_date"
        )
        assert "ds" not in frame.columns, (
            f"monthly_future_{label} must not contain the Prophet-specific 'ds' column"
        )


def test_generic_future_frames_do_not_contain_target():
    """Target column (monthly_demand) must be absent from all future frames."""
    f3, f6, f12 = _build_frames()
    for frame, label in ((f3, "3m"), (f6, "6m"), (f12, "12m")):
        assert "monthly_demand" not in frame.columns, (
            f"monthly_future_{label} must not contain the target column"
        )


def test_generic_future_frames_horizons_contain_correct_month_count():
    """3m/6m/12m frames cover exactly 3/6/12 future months (one SKU)."""
    f3, f6, f12 = _build_frames(n_history=24)
    assert len(f3) == 3   # noqa: PLR2004
    assert len(f6) == 6   # noqa: PLR2004
    assert len(f12) == 12  # noqa: PLR2004


def test_generic_future_frames_start_after_last_historical_month():
    """Every future frame must begin strictly after the last historical date."""
    modeling_data = _make_modeling_data(n=24)  # last date = 2024-12-01
    last_historical = modeling_data["month_start_date"].max()

    f3, f6, f12 = build_monthly_generic_future_frames(
        monthly_modeling_data=modeling_data,
        monthly_calendar_features=_make_calendar_features(40),
        monthly_exogenous_features=_make_exogenous_features(),
        parameters=_make_params(),
        calendar_parameters=_make_calendar_params(),
    )

    for frame, label in ((f3, "3m"), (f6, "6m"), (f12, "12m")):
        min_future = frame["month_start_date"].min()
        assert min_future > last_historical, (
            f"monthly_future_{label} starts at {min_future}, which is not after "
            f"the last historical month {last_historical}"
        )


def test_generic_future_frames_contain_active_regressors():
    """All active regressors from the monthly config appear in every future frame."""
    f3, f6, f12 = _build_frames(active_regressors=["business_days"])
    for frame, label in ((f3, "3m"), (f6, "6m"), (f12, "12m")):
        assert "business_days" in frame.columns, (
            f"monthly_future_{label} is missing active regressor 'business_days'"
        )


def test_generic_future_frames_contain_sku_column():
    """SKU column is present in every future frame."""
    f3, f6, f12 = _build_frames()
    for frame, label in ((f3, "3m"), (f6, "6m"), (f12, "12m")):
        assert "sku" in frame.columns, (
            f"monthly_future_{label} is missing the 'sku' column"
        )


def test_generic_future_frames_sku_values_match_modeling_data():
    """SKU values in future frames match the SKUs present in historical data."""
    f3, _, _ = _build_frames()
    assert set(f3["sku"].unique()) == {"SKU_001"}


def test_generic_future_frames_no_nulls_in_date_or_regressors():
    """No null values in date or regressor columns of any future frame."""
    f3, f6, f12 = _build_frames()
    for frame, label in ((f3, "3m"), (f6, "6m"), (f12, "12m")):
        null_counts = frame[["month_start_date", "business_days"]].isnull().sum()
        assert null_counts.sum() == 0, (
            f"monthly_future_{label} has null values: {null_counts[null_counts > 0].to_dict()}"
        )


def test_generic_future_frames_unknown_regressor_raises():
    """An active regressor absent from both calendar and exogenous sources raises ValueError."""
    with pytest.raises(ValueError, match="not available"):
        _build_frames(active_regressors=["nonexistent_feature_xyz"])


def test_generic_future_frames_multiple_skus_cross_joined():
    """When multiple SKUs are present, each SKU appears in every future month."""
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    modeling_data = pd.DataFrame(
        {
            "month_start_date": list(dates) * 2,
            "monthly_demand": np.linspace(100.0, 200.0, 48),
            "sku": ["SKU_001"] * 24 + ["SKU_002"] * 24,
            "business_days": [22] * 48,
        }
    )
    f3, _, _ = build_monthly_generic_future_frames(
        monthly_modeling_data=modeling_data,
        monthly_calendar_features=_make_calendar_features(40),
        monthly_exogenous_features=_make_exogenous_features(),
        parameters=_make_params(),
        calendar_parameters=_make_calendar_params(),
    )
    assert set(f3["sku"].unique()) == {"SKU_001", "SKU_002"}
    assert len(f3) == 6  # 2 SKUs × 3 months  # noqa: PLR2004
