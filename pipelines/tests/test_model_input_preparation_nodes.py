"""Unit tests for Monthly Prophet model-input preparation nodes."""

import pandas as pd
import pytest

from hdf_pipelines.pipelines.model_input_preparation.nodes import (
    build_monthly_prophet_future_regressors,
    prepare_monthly_prophet_modeling_data,
    split_monthly_prophet_data,
)

_EXPECTED_MODELING_ROWS = 3  # 4 input rows - 1 null regressor row = 3
_EXPECTED_TRAIN_ROWS = 3  # 5 rows - 1 validation month - 1 test month = 3
_EXPECTED_FULL_TRAIN_ROWS = 5  # all rows of the modeling DataFrame (no rows excluded)
_HORIZON_3M = 3
_HORIZON_6M = 6
_HORIZON_12M = 12


def _model_input_parameters() -> dict:
    """Return a minimal monthly Prophet parameter config for testing."""
    return {
        "monthly_prophet": {
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "sku_column": "sku",
            "prophet_date_column": "ds",
            "prophet_target_column": "y",
            "active_regressors": [
                "business_days",
                "pfizer_limited",
                "pfizer_limited_lag_1",
            ],
            "missing_values": {
                "drop_rows_with_null_target": True,
                "drop_rows_with_null_active_regressors": True,
            },
            "split": {
                "mode": "months",
                "validation_months": 1,
                "test_months": 1,
            },
            "future": {
                "horizons_months": [3, 6],
            },
        }
    }


def _calendar_parameters() -> dict:
    """Return a minimal calendar feature parameter config for testing."""
    return {
        "calendar_features": {
            "country_holidays": "CO",
            "observed_holidays": True,
            "weekmask": "Mon Tue Wed Thu Fri",
        }
    }


def _modeling_df_for_split_tests() -> pd.DataFrame:
    """Return a 15-month modeling DataFrame with one SKU for split tests."""
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=15, freq="MS"),
            "y": list(range(10, 25)),
            "sku": ["SKU-1"] * 15,
            "business_days": [20, 21, 22] * 5,
            "pfizer_limited": [0.0, 1.0, 0.0] * 5,
            "pfizer_limited_lag_1": [0.0, 0.0, 1.0] * 5,
        }
    )


def _split_metadata_stub() -> dict:
    """Return a pre-populated metadata dict to use as input for split tests."""
    return {
        "model_family": "prophet",
        "granularity": "monthly",
        "active_regressors": [
            "business_days",
            "pfizer_limited",
            "pfizer_limited_lag_1",
        ],
        "dropped_rows": {
            "null_target": 0,
            "null_active_regressors": 0,
        },
    }


def test_prepare_monthly_prophet_modeling_data_drops_null_regressor_rows():
    """Rows with null active regressors are dropped and counted in metadata."""
    feature_df = pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "sku": ["SKU-1"] * 4,
            "monthly_demand": [10.0, 12.0, 14.0, 16.0],
            "business_days": [22, 20, 21, 22],
            "pfizer_limited": [0.0, 1.0, 0.0, 0.0],
            "pfizer_limited_lag_1": [None, 0.0, 1.0, 0.0],
        }
    )

    modeling_df, metadata = prepare_monthly_prophet_modeling_data(
        feature_df,
        _model_input_parameters(),
    )

    assert list(modeling_df.columns) == [
        "ds",
        "y",
        "sku",
        "business_days",
        "pfizer_limited",
        "pfizer_limited_lag_1",
    ]
    assert len(modeling_df) == _EXPECTED_MODELING_ROWS
    assert modeling_df["ds"].min() == pd.Timestamp("2024-02-01")
    assert metadata["dropped_rows"]["null_target"] == 0
    assert metadata["dropped_rows"]["null_active_regressors"] == 1


def test_split_months_mode_produces_correct_partition_sizes():
    """Months-mode split assigns trailing months to validation/test and the rest to train."""
    modeling_df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "y": [10.0, 12.0, 14.0, 16.0, 18.0],
            "sku": ["SKU-1"] * 5,
            "business_days": [22, 20, 21, 22, 20],
            "pfizer_limited": [0.0, 1.0, 0.0, 0.0, 1.0],
            "pfizer_limited_lag_1": [0.0, 0.0, 1.0, 0.0, 0.0],
        }
    )
    metadata = _split_metadata_stub()

    train, validation, test, full_train, updated_metadata = split_monthly_prophet_data(
        modeling_df,
        metadata,
        _model_input_parameters(),
    )

    assert len(train) == _EXPECTED_TRAIN_ROWS
    assert len(validation) == 1
    assert len(test) == 1
    assert len(full_train) == _EXPECTED_FULL_TRAIN_ROWS
    assert updated_metadata["split_mode"] == "months"


def test_future_regressors_exclude_target_and_cover_configured_horizons():
    """Future regressor frames omit the target column and span the correct date range per horizon."""
    # modeling_df covers Jan–May 2024; future windows start at June 2024
    modeling_df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "y": [10.0, 12.0, 14.0, 16.0, 18.0],
            "sku": ["SKU-1"] * 5,
            "business_days": [22, 20, 21, 22, 20],
            "pfizer_limited": [0.0, 1.0, 0.0, 0.0, 1.0],
            "pfizer_limited_lag_1": [0.0, 0.0, 1.0, 0.0, 0.0],
        }
    )
    # calendar_df covers only the 5 historical months; the node generates future months
    calendar_df = pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "business_days": [22, 20, 21, 22, 20],
            "total_tuesdays": [5, 4, 4, 5, 4],
            "total_thursdays": [4, 5, 4, 4, 5],
            "working_tuesdays": [5, 4, 4, 5, 4],
            "working_thursdays": [4, 5, 4, 4, 5],
            "has_5_working_tuesdays": [1, 0, 0, 1, 0],
            "has_5_working_thursdays": [0, 1, 0, 0, 1],
            "tuesday_holidays": [0, 0, 0, 0, 0],
            "thursday_holidays": [0, 0, 0, 0, 0],
            "total_holidays": [1, 0, 1, 0, 1],
        }
    )
    # exogenous_df must cover up to May 2025 (last month of the 12-month horizon:
    # 5 historical months Jan–May 2024 + 12 future months Jun 2024–May 2025 = 17 months)
    _n_exogenous = 17
    exogenous_df = pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=_n_exogenous, freq="MS"),
            "pfizer_limited": ([0.0, 1.0, 0.0] * 6)[:_n_exogenous],
            "pfizer_limited_lag_1": ([0.0, 0.0, 1.0] * 6)[:_n_exogenous],
        }
    )

    params = _model_input_parameters()
    params["monthly_prophet"]["future"]["horizons_months"] = [3, 6, 12]

    future_3m, future_6m, future_12m = build_monthly_prophet_future_regressors(
        modeling_df,
        calendar_df,
        exogenous_df,
        params,
        _calendar_parameters(),
    )

    expected_columns = ["ds", "sku", "business_days", "pfizer_limited", "pfizer_limited_lag_1"]
    assert list(future_3m.columns) == expected_columns
    assert list(future_6m.columns) == expected_columns
    assert list(future_12m.columns) == expected_columns
    assert "y" not in future_3m.columns
    assert "y" not in future_6m.columns
    assert "y" not in future_12m.columns
    # last historical month = 2024-05-01 → future starts 2024-06-01
    assert future_3m["ds"].min() == pd.Timestamp("2024-06-01")
    assert future_3m["ds"].max() == pd.Timestamp("2024-08-01")   # 3 months: Jun–Aug 2024
    assert future_6m["ds"].max() == pd.Timestamp("2024-11-01")   # 6 months: Jun–Nov 2024
    assert future_12m["ds"].max() == pd.Timestamp("2025-05-01")  # 12 months: Jun 2024–May 2025
    assert len(future_3m) == _HORIZON_3M
    assert len(future_6m) == _HORIZON_6M
    assert len(future_12m) == _HORIZON_12M


def test_split_date_mode_respects_boundaries():
    """Date-mode split assigns rows to the correct partition according to explicit cutoff dates."""
    modeling_df = _modeling_df_for_split_tests()
    metadata = _split_metadata_stub()
    params = _model_input_parameters()
    params["monthly_prophet"]["split"] = {
        "mode": "date",
        "train_end_date": "2024-09-01",
        "validation_end_date": "2024-12-01",
        "test_end_date": "2025-03-01",
    }

    train, validation, test, full_train, updated_metadata = split_monthly_prophet_data(
        modeling_df,
        metadata,
        params,
    )

    assert train["ds"].max() == pd.Timestamp("2024-09-01")
    assert validation["ds"].min() == pd.Timestamp("2024-10-01")
    assert validation["ds"].max() == pd.Timestamp("2024-12-01")
    assert test["ds"].min() == pd.Timestamp("2025-01-01")
    assert test["ds"].max() == pd.Timestamp("2025-03-01")
    # test_end_date spans all 15 fixture rows, so full_train equals the entire dataset
    assert len(full_train) == len(modeling_df)
    assert updated_metadata["split_mode"] == "date"


def test_split_unsupported_mode_raises_clear_error():
    """An unrecognized split mode raises ValueError with a descriptive message."""
    params = _model_input_parameters()
    params["monthly_prophet"]["split"] = {"mode": "weekly-ish"}

    with pytest.raises(ValueError, match="Unsupported split mode"):
        split_monthly_prophet_data(
            _modeling_df_for_split_tests(),
            _split_metadata_stub(),
            params,
        )


def test_split_date_mode_rejects_invalid_order():
    """Date-mode split raises ValueError when cutoff dates are not strictly increasing."""
    params = _model_input_parameters()
    params["monthly_prophet"]["split"] = {
        "mode": "date",
        "train_end_date": "2024-12-01",
        "validation_end_date": "2024-09-01",
        "test_end_date": "2025-03-01",
    }

    with pytest.raises(
        ValueError,
        match="train_end_date < validation_end_date < test_end_date",
    ):
        split_monthly_prophet_data(
            _modeling_df_for_split_tests(),
            _split_metadata_stub(),
            params,
        )


def test_split_months_mode_rejects_empty_train():
    """Months-mode split raises ValueError when validation + test consume all available history."""
    params = _model_input_parameters()
    params["monthly_prophet"]["split"] = {
        "mode": "months",
        "validation_months": 8,
        "test_months": 8,
    }

    with pytest.raises(ValueError, match="Not enough historical months"):
        split_monthly_prophet_data(
            _modeling_df_for_split_tests(),
            _split_metadata_stub(),
            params,
        )
