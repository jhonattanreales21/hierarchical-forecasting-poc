"""Unit tests for monthly model-input preparation nodes."""

import pandas as pd
import pytest

from hdf_pipelines.pipelines.model_input_preparation.nodes import (
    adapt_monthly_data_for_prophet,
    adapt_monthly_data_for_sarimax,
    build_monthly_modeling_data,
    build_monthly_prophet_future_regressors,
    build_monthly_split_metadata,
    split_monthly_modeling_data,
)

_EXPECTED_MODELING_ROWS = 3  # 4 input rows - 1 null regressor row = 3
_EXPECTED_TRAIN_ROWS = 3  # 5 rows - 1 validation month - 1 test month = 3
_EXPECTED_FULL_TRAIN_ROWS = 5  # all rows of the modeling DataFrame (no rows excluded)
_HORIZON_3M = 3
_HORIZON_6M = 6
_HORIZON_12M = 12


def _model_input_parameters() -> dict:
    """Return a minimal parameter config for testing — includes both monthly and monthly_prophet."""
    return {
        "monthly": {
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "sku_column": "sku",
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
        },
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
        },
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


def _modeling_df_generic() -> pd.DataFrame:
    """Return a 15-month generic modeling DataFrame (month_start_date / monthly_demand)."""
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=15, freq="MS"),
            "monthly_demand": list(range(10, 25)),
            "sku": ["SKU-1"] * 15,
            "business_days": [20, 21, 22] * 5,
            "pfizer_limited": [0.0, 1.0, 0.0] * 5,
            "pfizer_limited_lag_1": [0.0, 0.0, 1.0] * 5,
        }
    )


def _preparation_metadata_stub() -> dict:
    """Return a pre-populated generic preparation metadata dict for split tests."""
    return {
        "granularity": "monthly",
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "sku_column": "sku",
        "active_regressors": [
            "business_days",
            "pfizer_limited",
            "pfizer_limited_lag_1",
        ],
        "dropped_rows": {
            "null_target": 0,
            "null_active_regressors": 0,
        },
        "modeling_data": {"start_date": "2024-01-01", "end_date": "2025-03-01", "rows": 15},
    }


# ── Generic modeling data ──────────────────────────────────────────────────────

def test_build_monthly_modeling_data_drops_null_regressor_rows():
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

    modeling_df, metadata = build_monthly_modeling_data(
        feature_df,
        _model_input_parameters(),
    )

    assert list(modeling_df.columns) == [
        "month_start_date",
        "monthly_demand",
        "sku",
        "business_days",
        "pfizer_limited",
        "pfizer_limited_lag_1",
    ]
    assert "ds" not in modeling_df.columns
    assert "y" not in modeling_df.columns
    assert len(modeling_df) == _EXPECTED_MODELING_ROWS
    assert modeling_df["month_start_date"].min() == pd.Timestamp("2024-02-01")
    assert metadata["dropped_rows"]["null_target"] == 0
    assert metadata["dropped_rows"]["null_active_regressors"] == 1
    assert metadata["granularity"] == "monthly"
    assert metadata["date_column"] == "month_start_date"
    assert metadata["target_column"] == "monthly_demand"


def test_build_monthly_modeling_data_uses_generic_column_names():
    """Output must keep month_start_date / monthly_demand — never ds / y."""
    feature_df = pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=3, freq="MS"),
            "sku": ["SKU-1"] * 3,
            "monthly_demand": [10.0, 12.0, 14.0],
            "business_days": [22, 20, 21],
            "pfizer_limited": [0.0, 1.0, 0.0],
            "pfizer_limited_lag_1": [0.0, 0.0, 1.0],
        }
    )

    modeling_df, _ = build_monthly_modeling_data(feature_df, _model_input_parameters())

    assert "month_start_date" in modeling_df.columns
    assert "monthly_demand" in modeling_df.columns
    assert "ds" not in modeling_df.columns
    assert "y" not in modeling_df.columns


# ── Generic temporal splits ────────────────────────────────────────────────────

def test_split_months_mode_produces_correct_partition_sizes():
    """Months-mode split assigns trailing months to validation/test and the rest to train."""
    modeling_df = pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "monthly_demand": [10.0, 12.0, 14.0, 16.0, 18.0],
            "sku": ["SKU-1"] * 5,
            "business_days": [22, 20, 21, 22, 20],
            "pfizer_limited": [0.0, 1.0, 0.0, 0.0, 1.0],
            "pfizer_limited_lag_1": [0.0, 0.0, 1.0, 0.0, 0.0],
        }
    )
    metadata = _preparation_metadata_stub()

    train, validation, test, full_train, updated_metadata = split_monthly_modeling_data(
        modeling_df,
        metadata,
        _model_input_parameters(),
    )

    assert len(train) == _EXPECTED_TRAIN_ROWS
    assert len(validation) == 1
    assert len(test) == 1
    assert len(full_train) == _EXPECTED_FULL_TRAIN_ROWS
    assert updated_metadata["split_mode"] == "months"


def test_split_monthly_modeling_data_splits_are_non_overlapping():
    """Train, validation, and test date ranges must not overlap."""
    train, validation, test, _, _ = split_monthly_modeling_data(
        _modeling_df_generic(),
        _preparation_metadata_stub(),
        _model_input_parameters(),
    )

    assert train["month_start_date"].max() < validation["month_start_date"].min()
    assert validation["month_start_date"].max() < test["month_start_date"].min()


def test_split_date_mode_respects_boundaries():
    """Date-mode split assigns rows to the correct partition according to explicit cutoff dates."""
    metadata = _preparation_metadata_stub()
    params = _model_input_parameters()
    params["monthly"]["split"] = {
        "mode": "date",
        "train_end_date": "2024-09-01",
        "validation_end_date": "2024-12-01",
        "test_end_date": "2025-03-01",
    }

    train, validation, test, full_train, updated_metadata = split_monthly_modeling_data(
        _modeling_df_generic(),
        metadata,
        params,
    )

    assert train["month_start_date"].max() == pd.Timestamp("2024-09-01")
    assert validation["month_start_date"].min() == pd.Timestamp("2024-10-01")
    assert validation["month_start_date"].max() == pd.Timestamp("2024-12-01")
    assert test["month_start_date"].min() == pd.Timestamp("2025-01-01")
    assert test["month_start_date"].max() == pd.Timestamp("2025-03-01")
    assert len(full_train) == len(_modeling_df_generic())
    assert updated_metadata["split_mode"] == "date"


def test_split_unsupported_mode_raises_clear_error():
    """An unrecognized split mode raises ValueError with a descriptive message."""
    params = _model_input_parameters()
    params["monthly"]["split"] = {"mode": "weekly-ish"}

    with pytest.raises(ValueError, match="Unsupported split mode"):
        split_monthly_modeling_data(
            _modeling_df_generic(),
            _preparation_metadata_stub(),
            params,
        )


def test_split_date_mode_rejects_invalid_order():
    """Date-mode split raises ValueError when cutoff dates are not strictly increasing."""
    params = _model_input_parameters()
    params["monthly"]["split"] = {
        "mode": "date",
        "train_end_date": "2024-12-01",
        "validation_end_date": "2024-09-01",
        "test_end_date": "2025-03-01",
    }

    with pytest.raises(
        ValueError,
        match="train_end_date < validation_end_date < test_end_date",
    ):
        split_monthly_modeling_data(
            _modeling_df_generic(),
            _preparation_metadata_stub(),
            params,
        )


def test_split_months_mode_rejects_empty_train():
    """Months-mode split raises ValueError when validation + test consume all available history."""
    params = _model_input_parameters()
    params["monthly"]["split"] = {
        "mode": "months",
        "validation_months": 8,
        "test_months": 8,
    }

    with pytest.raises(ValueError, match="Not enough historical months"):
        split_monthly_modeling_data(
            _modeling_df_generic(),
            _preparation_metadata_stub(),
            params,
        )


# ── Generic split metadata ─────────────────────────────────────────────────────

def test_build_monthly_split_metadata_has_expected_fields():
    """build_monthly_split_metadata must emit all required generic metadata fields."""
    metadata_stub = dict(_preparation_metadata_stub())
    metadata_stub["split_mode"] = "months"
    modeling_df = _modeling_df_generic()
    params = _model_input_parameters()

    train, validation, test, full_train, extended_metadata = split_monthly_modeling_data(
        modeling_df, metadata_stub, params
    )

    result = build_monthly_split_metadata(
        train, validation, test, full_train, extended_metadata, params
    )

    assert result["granularity"] == "monthly"
    assert result["date_column"] == "month_start_date"
    assert result["target_column"] == "monthly_demand"
    assert result["split_mode"] == "months"
    assert "active_features" in result
    assert "train" in result
    assert "validation" in result
    assert "test" in result
    assert "full_train" in result
    assert "row_counts" in result
    assert result["row_counts"]["full_train"] == len(modeling_df)  # all 15 rows: train + validation + test
    assert result["created_by"] == "model_input_preparation"


# ── Prophet compatibility adapter ──────────────────────────────────────────────

def test_adapt_monthly_data_for_prophet_renames_columns():
    """Adapter must rename month_start_date → ds and monthly_demand → y."""
    modeling_df = _modeling_df_generic()
    params = _model_input_parameters()

    train, validation, test, full_train, split_meta = split_monthly_modeling_data(
        modeling_df, _preparation_metadata_stub(), params
    )
    generic_meta = build_monthly_split_metadata(
        train, validation, test, full_train,
        {**_preparation_metadata_stub(), "split_mode": "months",
         "train": {}, "validation": {}, "test": {}, "full_train": {}},
        params,
    )

    (
        prophet_modeling_data,
        prophet_train,
        prophet_validation,
        prophet_test,
        prophet_full_train,
        adapter_meta,
    ) = adapt_monthly_data_for_prophet(
        modeling_df, train, validation, test, full_train, generic_meta, params
    )

    for df in [prophet_modeling_data, prophet_train, prophet_validation, prophet_test, prophet_full_train]:
        assert "ds" in df.columns
        assert "y" in df.columns
        assert "month_start_date" not in df.columns
        assert "monthly_demand" not in df.columns

    assert adapter_meta["model_family"] == "prophet"
    assert adapter_meta["granularity"] == "monthly"


def test_adapt_monthly_data_for_prophet_preserves_row_counts():
    """Adapter must not drop or duplicate rows."""
    modeling_df = _modeling_df_generic()
    params = _model_input_parameters()

    train, validation, test, full_train, split_meta = split_monthly_modeling_data(
        modeling_df, _preparation_metadata_stub(), params
    )
    generic_meta = build_monthly_split_metadata(
        train, validation, test, full_train,
        {**_preparation_metadata_stub(), "split_mode": "months",
         "train": {}, "validation": {}, "test": {}, "full_train": {}},
        params,
    )

    prophet_modeling_data, prophet_train, prophet_validation, prophet_test, prophet_full_train, _ = (
        adapt_monthly_data_for_prophet(
            modeling_df, train, validation, test, full_train, generic_meta, params
        )
    )

    assert len(prophet_modeling_data) == len(modeling_df)
    assert len(prophet_train) == len(train)
    assert len(prophet_validation) == len(validation)
    assert len(prophet_test) == len(test)
    assert len(prophet_full_train) == len(full_train)


def test_adapt_monthly_data_for_prophet_preserves_regressors():
    """Active regressor columns must survive the rename unchanged."""
    modeling_df = _modeling_df_generic()
    params = _model_input_parameters()

    train, validation, test, full_train, split_meta = split_monthly_modeling_data(
        modeling_df, _preparation_metadata_stub(), params
    )
    generic_meta = build_monthly_split_metadata(
        train, validation, test, full_train,
        {**_preparation_metadata_stub(), "split_mode": "months",
         "train": {}, "validation": {}, "test": {}, "full_train": {}},
        params,
    )

    result = adapt_monthly_data_for_prophet(
        modeling_df, train, validation, test, full_train, generic_meta, params
    )
    prophet_train_df = result[1]

    active_regressors = params["monthly"]["active_regressors"]
    for col in active_regressors:
        assert col in prophet_train_df.columns


# ── Prophet future regressors (unchanged behavior) ────────────────────────────

def test_future_regressors_exclude_target_and_cover_configured_horizons():
    """Future regressor frames omit the target column and span the correct date range per horizon."""
    # modeling_df uses Prophet ds/y convention (input to build_monthly_prophet_future_regressors)
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
    _n_exogenous = 17
    exogenous_df = pd.DataFrame(
        {
            "month_start_date": pd.date_range(
                "2024-01-01", periods=_n_exogenous, freq="MS"
            ),
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

    expected_columns = [
        "ds",
        "sku",
        "business_days",
        "pfizer_limited",
        "pfizer_limited_lag_1",
    ]
    assert list(future_3m.columns) == expected_columns
    assert list(future_6m.columns) == expected_columns
    assert list(future_12m.columns) == expected_columns
    assert "y" not in future_3m.columns
    assert "y" not in future_6m.columns
    assert "y" not in future_12m.columns
    assert future_3m["ds"].min() == pd.Timestamp("2024-06-01")
    assert future_3m["ds"].max() == pd.Timestamp("2024-08-01")
    assert future_6m["ds"].max() == pd.Timestamp("2024-11-01")
    assert future_12m["ds"].max() == pd.Timestamp("2025-05-01")
    assert len(future_3m) == _HORIZON_3M
    assert len(future_6m) == _HORIZON_6M
    assert len(future_12m) == _HORIZON_12M


# ── SARIMAX adapter ────────────────────────────────────────────────────────────

def _sarimax_params() -> dict:
    """Minimal SARIMAX parameter block with no exogenous columns."""
    return {
        "monthly_sarimax": {
            "enabled": True,
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "sku_column": "sku",
            "frequency": "MS",
            "exogenous_columns": [],
            "allow_empty_exog": True,
            "require_regular_frequency": True,
            "sort_by_date": True,
            "drop_rows_with_null_target": True,
            "drop_rows_with_null_exog": False,
            "impute_exog": False,
            "output_format": "tabular",
        },
    }


def _make_generic_splits() -> tuple:
    """Return (train, validation, test, full_train, split_metadata) for SARIMAX tests."""
    modeling_df = pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "monthly_demand": [10.0, 12.0, 14.0, 16.0, 18.0],
            "sku": ["SKU-1"] * 5,
            "business_days": [22, 20, 21, 22, 20],
            "pfizer_limited": [0.0, 1.0, 0.0, 0.0, 1.0],
        }
    )
    train = modeling_df.iloc[:3].copy().reset_index(drop=True)
    validation = modeling_df.iloc[3:4].copy().reset_index(drop=True)
    test = modeling_df.iloc[4:5].copy().reset_index(drop=True)
    full_train = modeling_df.copy()
    meta = {
        "granularity": "monthly",
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "sku_column": "sku",
        "split_mode": "months",
        "active_features": ["business_days", "pfizer_limited"],
        "dropped_rows": {"null_target": 0, "null_active_regressors": 0},
        "created_by": "model_input_preparation",
        "train": {"rows": 3},
        "validation": {"rows": 1},
        "test": {"rows": 1},
        "full_train": {"rows": 5},
    }
    return train, validation, test, full_train, meta


def test_sarimax_adapter_creates_all_five_outputs():
    """Adapter returns four DataFrames and a metadata dict."""
    train, validation, test, full_train, meta = _make_generic_splits()

    result = adapt_monthly_data_for_sarimax(train, validation, test, full_train, meta, _sarimax_params())

    assert len(result) == 5
    for item in result[:4]:
        assert isinstance(item, pd.DataFrame)
    assert isinstance(result[4], dict)


def test_sarimax_adapter_output_columns_exclude_sku_and_unused_regressors():
    """Output keeps only [date_column, target_column]; sku and other regressors are dropped."""
    train, validation, test, full_train, meta = _make_generic_splits()

    sarimax_train, *_ = adapt_monthly_data_for_sarimax(
        train, validation, test, full_train, meta, _sarimax_params()
    )

    assert list(sarimax_train.columns) == ["month_start_date", "monthly_demand"]
    assert "sku" not in sarimax_train.columns
    assert "business_days" not in sarimax_train.columns


def test_sarimax_adapter_raises_on_missing_required_column():
    """Adapter raises ValueError with a clear message when date or target column is absent."""
    train, validation, test, full_train, meta = _make_generic_splits()

    with pytest.raises(ValueError, match="Missing required columns"):
        adapt_monthly_data_for_sarimax(
            train.drop(columns=["month_start_date"]), validation, test, full_train, meta, _sarimax_params()
        )

    with pytest.raises(ValueError, match="Missing required columns"):
        adapt_monthly_data_for_sarimax(
            train.drop(columns=["monthly_demand"]), validation, test, full_train, meta, _sarimax_params()
        )


def test_sarimax_adapter_null_target_dropped_and_counted_in_metadata():
    """Null target rows are dropped and null_target_rows_dropped is recorded in metadata."""
    train, validation, test, full_train, meta = _make_generic_splits()
    train_with_null = train.copy()
    train_with_null.loc[0, "monthly_demand"] = None
    params = _sarimax_params()

    _, _, _, _, sarimax_meta = adapt_monthly_data_for_sarimax(
        train_with_null, validation, test, full_train, meta, params
    )

    assert sarimax_meta["splits"]["train"]["null_target_rows_dropped"] == 1
    assert sarimax_meta["splits"]["train"]["rows"] == len(train) - 1


def test_sarimax_adapter_metadata_contract():
    """sarimax_split_metadata contains model_family, granularity, frequency, and per-split diagnostics."""
    train, validation, test, full_train, meta = _make_generic_splits()

    _, _, _, _, sarimax_meta = adapt_monthly_data_for_sarimax(
        train, validation, test, full_train, meta, _sarimax_params()
    )

    assert sarimax_meta["model_family"] == "sarimax"
    assert sarimax_meta["granularity"] == "monthly"
    assert sarimax_meta["frequency"] == "MS"
    assert sarimax_meta["date_column"] == "month_start_date"
    assert sarimax_meta["target_column"] == "monthly_demand"
    for split_name in ("train", "validation", "test", "full_train"):
        info = sarimax_meta["splits"][split_name]
        assert {"rows", "start", "end", "missing_periods"} <= info.keys()
    assert sarimax_meta["created_by"] == "model_input_preparation.sarimax_adapter"
