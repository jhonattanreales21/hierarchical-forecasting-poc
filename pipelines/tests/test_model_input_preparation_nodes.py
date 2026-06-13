"""Unit tests for monthly model-input preparation nodes (rolling-origin protocol).

The fixed train/validation/test hold-out was removed: the series is kept whole and
the rolling-origin engine slices cycles at train time. These tests cover the
full-history collapse, the rolling-origin window invariants (last cycle predicts
[L-2, L-1, L]), and the Prophet/SARIMAX adapters.
"""

import pandas as pd
import pytest

from hdf_pipelines.pipelines.model_input_preparation.nodes import (
    adapt_monthly_data_for_catboost,
    adapt_monthly_data_for_prophet,
    adapt_monthly_data_for_sarimax,
    build_monthly_modeling_data,
    build_monthly_rolling_origin_windows,
    build_monthly_split_metadata,
    prepare_monthly_full_history,
)

_HORIZON = 3
_N_CYCLES = 5


def _model_input_parameters() -> dict:
    """Minimal parameter config with the rolling-origin block (no split block)."""
    rolling_origin = {
        "horizon": _HORIZON,
        "n_cycles": _N_CYCLES,
        "window": "expanding",
        "step_months": 1,
        "min_train_periods": 12,
    }
    regressors = ["business_days", "pfizer_limited", "pfizer_limited_lag_1"]
    missing = {
        "drop_rows_with_null_target": True,
        "drop_rows_with_null_active_regressors": True,
    }
    return {
        "monthly": {
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "sku_column": "sku",
            "active_regressors": regressors,
            "missing_values": missing,
            "rolling_origin": rolling_origin,
        },
        "monthly_prophet": {
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "sku_column": "sku",
            "prophet_date_column": "ds",
            "prophet_target_column": "y",
            "active_regressors": regressors,
            "missing_values": missing,
            "future": {"horizons_months": [3, 6]},
        },
        "monthly_sarimax": {
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "sku_column": "sku",
            "frequency": "MS",
            "active_regressors": ["pfizer_limited", "pfizer_limited_lag_1"],
            "exogenous_columns": ["pfizer_limited", "pfizer_limited_lag_1"],
            "allow_empty_exog": True,
            "require_regular_frequency": True,
            "sort_by_date": True,
            "drop_rows_with_null_target": True,
            "output_format": "tabular",
        },
        "monthly_catboost": {
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "sku_column": "sku",
            "active_regressors": regressors,
            "target_lags": [1, 2],
            "rolling_windows": [3],
            "include_rolling_std": True,
            "include_rolling_min_max": False,
            "trend_diffs": [1],
            "trend_pct_changes": [1],
            "drop_rows_with_null_target_features": True,
            "include_missingness_flags": False,
            "missingness_flag_columns": [],
        },
    }


def _modeling_df_generic(n: int = 24) -> pd.DataFrame:
    """Return an n-month generic modeling DataFrame (sorted, contiguous)."""
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=n, freq="MS"),
            "monthly_demand": [float(100 + 5 * i) for i in range(n)],
            "sku": ["SKU-1"] * n,
            "business_days": [20, 21, 22] * (n // 3),
            "pfizer_limited": [0.0, 1.0, 0.0] * (n // 3),
            "pfizer_limited_lag_1": [0.0, 0.0, 1.0] * (n // 3),
        }
    )


def _preparation_metadata_stub() -> dict:
    return {
        "granularity": "monthly",
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "sku_column": "sku",
        "active_regressors": [
            "business_days", "pfizer_limited", "pfizer_limited_lag_1",
        ],
        "dropped_rows": {"null_target": 0, "null_active_regressors": 0},
        "modeling_data": {"start_date": "2024-01-01", "end_date": "2025-12-01", "rows": 24},
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
        feature_df, _model_input_parameters()
    )
    assert len(modeling_df) == 3
    assert metadata["dropped_rows"]["null_active_regressors"] == 1


def test_family_regressor_outside_monthly_pool_raises():
    """Family-specific regressors must be selected from the monthly feature pool."""
    feature_df = _modeling_df_generic(6)
    params = _model_input_parameters()
    params["monthly_prophet"]["active_regressors"] = ["business_days", "unknown_reg"]

    with pytest.raises(ValueError, match="outside"):
        build_monthly_modeling_data(feature_df, params)


# ── Full history + rolling-origin windows ─────────────────────────────────────


def test_prepare_monthly_full_history_sorts_and_keeps_all_rows():
    modeling_df = _modeling_df_generic(24).sample(frac=1.0, random_state=0)
    full_train, meta = prepare_monthly_full_history(
        modeling_df, _preparation_metadata_stub(), _model_input_parameters()
    )
    assert len(full_train) == 24
    assert full_train["month_start_date"].is_monotonic_increasing
    assert meta["full_train"]["rows"] == 24


def test_rolling_origin_windows_last_cycle_predicts_final_months():
    full_train = _modeling_df_generic(24)
    windows = build_monthly_rolling_origin_windows(
        full_train, _model_input_parameters()
    )
    assert windows["n_cycles"] == _N_CYCLES
    assert windows["horizon"] == _HORIZON
    assert windows["window"] == "expanding"
    # L = 2025-12-01; last cycle targets [L-2, L-1, L].
    assert windows["last_observed_month"] == "2025-12-01"
    assert windows["cycles"][-1]["target_dates"] == [
        "2025-10-01", "2025-11-01", "2025-12-01",
    ]


def test_rolling_origin_windows_raise_when_series_too_short():
    params = _model_input_parameters()
    params["monthly"]["rolling_origin"]["min_train_periods"] = 6
    short = _modeling_df_generic(6)  # too few months for 5 cycles, H=3
    with pytest.raises(ValueError):
        build_monthly_rolling_origin_windows(short, params)


def test_build_monthly_split_metadata_is_rolling_origin():
    full_train = _modeling_df_generic(24)
    meta = build_monthly_split_metadata(
        full_train, _preparation_metadata_stub(), _model_input_parameters()
    )
    assert meta["evaluation_mode"] == "rolling_origin"
    assert meta["row_counts"]["full_train"] == 24
    assert "validation" not in meta
    assert "test" not in meta


# ── Adapters ──────────────────────────────────────────────────────────────────


def test_adapt_monthly_data_for_prophet_renames_and_collapses():
    params = _model_input_parameters()
    full_train = _modeling_df_generic(24)
    split_meta = build_monthly_split_metadata(
        full_train, _preparation_metadata_stub(), params
    )
    modeling_data, prophet_full_train, adapter_meta = adapt_monthly_data_for_prophet(
        full_train, full_train, split_meta, params
    )
    assert {"ds", "y"}.issubset(prophet_full_train.columns)
    assert "monthly_demand" not in prophet_full_train.columns
    assert len(prophet_full_train) == 24
    assert adapter_meta["evaluation_mode"] == "rolling_origin"
    assert {"ds", "y"}.issubset(modeling_data.columns)


def test_family_specific_regressor_subsets_are_applied_by_adapters():
    """Prophet, SARIMAX, and CatBoost can each consume a different regressor subset."""
    params = _model_input_parameters()
    params["monthly_prophet"]["active_regressors"] = ["business_days"]
    params["monthly_sarimax"]["active_regressors"] = ["pfizer_limited_lag_1"]
    params["monthly_sarimax"]["exogenous_columns"] = ["pfizer_limited_lag_1"]
    params["monthly_catboost"]["active_regressors"] = ["pfizer_limited"]

    full_train = _modeling_df_generic(24)
    split_meta = build_monthly_split_metadata(
        full_train, _preparation_metadata_stub(), params
    )

    _, prophet_full_train, prophet_meta = adapt_monthly_data_for_prophet(
        full_train, full_train, split_meta, params
    )
    sarimax_full_train, sarimax_meta = adapt_monthly_data_for_sarimax(
        full_train, split_meta, params
    )
    catboost_full_train, catboost_meta = adapt_monthly_data_for_catboost(
        full_train, split_meta, params
    )

    assert prophet_meta["active_regressors"] == ["business_days"]
    assert "business_days" in prophet_full_train.columns
    assert "pfizer_limited" not in prophet_full_train.columns
    assert sarimax_meta["exogenous_columns"] == ["pfizer_limited_lag_1"]
    assert "pfizer_limited_lag_1" in sarimax_full_train.columns
    assert "pfizer_limited" not in sarimax_full_train.columns
    assert catboost_meta["base_feature_columns"] == ["pfizer_limited"]
    assert "pfizer_limited" in catboost_full_train.columns
    assert "business_days" not in catboost_full_train.columns


def test_adapt_monthly_data_for_sarimax_returns_tabular_full_history():
    params = _model_input_parameters()
    full_train = _modeling_df_generic(24)
    split_meta = build_monthly_split_metadata(
        full_train, _preparation_metadata_stub(), params
    )
    sarimax_full_train, sarimax_meta = adapt_monthly_data_for_sarimax(
        full_train, split_meta, params
    )
    assert "monthly_demand" in sarimax_full_train.columns
    assert "pfizer_limited" in sarimax_full_train.columns
    assert len(sarimax_full_train) == 24
    assert sarimax_meta["model_family"] == "sarimax"
    assert "full_train" in sarimax_meta["splits"]
