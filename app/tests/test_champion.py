"""Unit tests for the model-family-agnostic champion helpers.

These tests cover the pure logic that lets the app display the current monthly
production champion without assuming Prophet, and the interval-rendering gate
that keeps prediction-interval bands out of the UI for champions that do not
produce them.
"""

import pandas as pd

from utils.champion import (
    extract_champion_identity,
    family_label,
    forecast_has_intervals,
    standardize_champion_metadata,
    standardize_forecast_columns,
)


def _generic_forecast() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-07-01", "2026-08-01"],
            "forecast": [1800.0, 1700.0, 1450.0],
            "forecast_lower": [1700.0, 1600.0, 1350.0],
            "forecast_upper": [1900.0, 1800.0, 1550.0],
        }
    )


def test_standardize_forecast_columns_renames_generic_schema():
    out = standardize_forecast_columns(_generic_forecast())
    assert {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(out.columns)
    assert pd.api.types.is_datetime64_any_dtype(out["ds"])


def test_standardize_champion_metadata_derives_precision_and_flag():
    meta = standardize_champion_metadata({"metrics": {"wape": 0.10}})
    assert meta["test_metrics"]["forecast_precision"] == 0.90
    assert meta["business_success_flag"] is True
    assert meta["business_success_precision_threshold"] == 0.85


def test_standardize_champion_metadata_flags_failure_above_threshold():
    meta = standardize_champion_metadata({"metrics": {"wape": 0.30}})
    assert meta["business_success_flag"] is False


def test_standardize_champion_metadata_handles_empty():
    assert standardize_champion_metadata({}) == {}


def test_extract_identity_prefers_generated_at_over_legacy_key():
    inf = {
        "model_family": "prophet",
        "champion_id": "prophet_candidate_079",
        "forecast_generated_at": "2026-06-03T05:40:41+00:00",
        "supported_horizons": [3, 6, 12],
        "has_prediction_interval": True,
        "interval_method": "prophet_native",
        "run_id": "run_abc",
    }
    meta = {
        "model_family": "prophet",
        "champion_id": "prophet_candidate_079",
        "selection": {"primary_metric": "wape", "selected_at": "2026-06-03T05:40:40Z"},
        "metrics": {"wape": 0.095, "mase": 0.46, "rmse": 95.8, "bias": -0.095},
        "test_period": {"start_date": "2026-02-01", "end_date": "2026-05-01"},
        "training_cutoff": "2026-05-01",
    }
    identity = extract_champion_identity(standardize_champion_metadata(meta), inf)
    assert identity["model_family"] == "prophet"
    assert identity["forecast_generated_at"] == "2026-06-03T05:40:41+00:00"
    assert identity["selection_metric"] == "wape"
    assert identity["selection_metric_value"] == 0.095
    assert identity["supported_horizons"] == [3, 6, 12]
    assert identity["has_prediction_interval"] is True


def test_extract_identity_is_family_agnostic_for_catboost():
    """A CatBoost champion with no intervals must still resolve cleanly."""
    meta = standardize_champion_metadata(
        {
            "model_family": "catboost",
            "champion_id": "catboost_trial_018",
            "metrics": {"wape": 0.22, "mase": 1.05, "rmse": 230.0, "bias": -0.2},
        }
    )
    inf = {
        "model_family": "catboost",
        "champion_id": "catboost_trial_018",
        "has_prediction_interval": False,
        "interval_method": None,
        "horizons": {"3": {}, "6": {}, "12": {}},
    }
    summary = pd.DataFrame(
        [{"production_champion_family": "catboost", "primary_metric": "wape"}]
    )
    identity = extract_champion_identity(meta, inf, summary)
    assert identity["model_family"] == "catboost"
    assert identity["has_prediction_interval"] is False
    assert identity["supported_horizons"] == [3, 6, 12]


def test_extract_identity_falls_back_to_selection_summary():
    summary = pd.DataFrame(
        [
            {
                "production_champion_family": "sarimax",
                "production_champion_id": "sarimax_trial_060",
                "primary_metric": "wape",
                "primary_metric_value": 0.11,
            }
        ]
    )
    identity = extract_champion_identity({}, {}, summary)
    assert identity["model_family"] == "sarimax"
    assert identity["champion_id"] == "sarimax_trial_060"
    assert identity["selection_metric_value"] == 0.11


def test_extract_identity_handles_all_empty():
    identity = extract_champion_identity({}, {}, None)
    assert identity["model_family"] is None
    assert identity["test_metrics"] == {}
    assert identity["supported_horizons"] == []


def test_forecast_has_intervals_true_for_real_bands():
    fc = standardize_forecast_columns(_generic_forecast())
    assert forecast_has_intervals(fc, {"has_prediction_interval": True}) is True


def test_forecast_has_intervals_false_when_metadata_denies():
    fc = standardize_forecast_columns(_generic_forecast())
    assert forecast_has_intervals(fc, {"has_prediction_interval": False}) is False


def test_forecast_has_intervals_false_for_zero_width_band():
    fc = standardize_forecast_columns(_generic_forecast())
    fc["yhat_lower"] = fc["yhat"]
    fc["yhat_upper"] = fc["yhat"]
    assert forecast_has_intervals(fc, {"has_prediction_interval": True}) is False


def test_forecast_has_intervals_false_for_missing_bounds():
    fc = pd.DataFrame({"ds": pd.to_datetime(["2026-06-01"]), "yhat": [1800.0]})
    assert forecast_has_intervals(fc, {"has_prediction_interval": True}) is False


def test_forecast_has_intervals_false_for_empty_frame():
    assert forecast_has_intervals(pd.DataFrame(), {}) is False


def test_forecast_has_intervals_false_when_all_null():
    fc = standardize_forecast_columns(_generic_forecast())
    fc["yhat_lower"] = None
    fc["yhat_upper"] = None
    assert forecast_has_intervals(fc, {"has_prediction_interval": True}) is False


def test_family_label_mapping_and_fallback():
    assert family_label("prophet") == "Prophet"
    assert family_label("sarimax") == "SARIMAX"
    assert family_label("catboost") == "CatBoost"
    assert family_label("nhits") == "nhits"
    assert family_label(None) == "production champion"
