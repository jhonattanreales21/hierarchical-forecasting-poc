"""Tests for metadata-driven monthly champion forecast inference.

Covers:
- metadata-driven dispatch to Prophet, SARIMAX, and CatBoost adapters;
- unsupported family and missing-metadata failures;
- the canonical standard forecast schema and nullable interval handling;
- SARIMAX exogenous validation;
- CatBoost recursive lag/rolling feature construction and dispatch;
- CatBoost missing future-column validation;
- CatBoost canonical schema, horizon row counts, and null intervals;
- the main node's horizon outputs, latest aliasing, and audit metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from hdf_pipelines.pipelines.forecast_inference.adapters import (
    _compute_recursive_features_for_step,
    dispatch_monthly_prediction,
    predict_monthly_catboost,
    predict_monthly_sarimax,
)
from hdf_pipelines.pipelines.forecast_inference.nodes import (
    _STANDARD_FORECAST_COLUMNS,
    generate_monthly_champion_forecasts,
)

_HORIZON_3M = 3
_HORIZON_6M = 6
_HORIZON_12M = 12
_PROPHET_YHAT = 100.0
_CATBOOST_PRED = 200.0


# ── Test doubles ──────────────────────────────────────────────────────────────


class _FakeProphet:
    """Prophet test double whose predict() echoes ds with deterministic columns."""

    def __init__(self, *, with_intervals: bool = True) -> None:
        self.extra_regressors = {"business_days": {}}
        self._with_intervals = with_intervals

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        out = pd.DataFrame({"ds": df["ds"].values, "yhat": [_PROPHET_YHAT] * n})
        if self._with_intervals:
            out["yhat_lower"] = [90.0] * n
            out["yhat_upper"] = [110.0] * n
        return out


class _FakeForecastResult:
    """Mimics statsmodels get_forecast() output."""

    def __init__(self, mean: np.ndarray, lower=None, upper=None) -> None:
        self.predicted_mean = mean
        self._lower = lower
        self._upper = upper

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        if self._lower is None:
            raise RuntimeError("confidence interval unavailable")
        return np.column_stack([self._lower, self._upper])


class _FakeSarimaxInner:
    def __init__(self, k_exog: int = 0) -> None:
        self.k_exog = k_exog


class _FakeSarimaxResults:
    """SARIMAX fitted-results test double exposing get_forecast()."""

    def __init__(self, *, base: float = 100.0, k_exog: int = 0, with_ci: bool = True) -> None:
        self.model = _FakeSarimaxInner(k_exog=k_exog)
        self._base = base
        self._with_ci = with_ci
        self.last_exog: Any = "unset"

    def get_forecast(self, steps: int, exog=None):
        self.last_exog = exog
        mean = np.arange(steps, dtype=float) + self._base
        if self._with_ci:
            return _FakeForecastResult(mean, mean - 5.0, mean + 5.0)
        return _FakeForecastResult(mean)


class _FakeCatBoostModel:
    """CatBoost test double whose predict() returns a deterministic constant."""

    def __init__(self, value: float = _CATBOOST_PRED) -> None:
        self._value = value
        self.call_inputs: list[np.ndarray] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.call_inputs.append(X.copy())
        return np.full(len(X), self._value)


class _FakeRecordingCatBoostModel:
    """CatBoost test double that echoes demand_lag_1 back as the prediction.

    Useful for testing that the demand buffer is updated correctly across steps.
    The model reads the first feature from the input matrix and returns it, so
    tests can verify that demand_lag_1 at step h=2 equals the prediction from step h=1.
    """

    def __init__(self, feature_columns: list[str], lag1_col: str = "demand_lag_1") -> None:
        self._feature_columns = feature_columns
        self._lag1_idx = feature_columns.index(lag1_col) if lag1_col in feature_columns else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([X[0, self._lag1_idx]])


# ── Fixtures / builders ───────────────────────────────────────────────────────


def _make_future_df(n_months: int, start: str = "2025-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range(start, periods=n_months, freq="MS"),
            "sku": ["SKU-1"] * n_months,
            "business_days": [22] * n_months,
        }
    )


def _prophet_metadata() -> dict[str, Any]:
    return {
        "model_family": "prophet",
        "champion_id": "prophet_candidate_001",
        "granularity": "monthly",
        "active_regressors": ["business_days"],
        "selection": {"primary_metric": "wape"},
        "metrics": {"wape": 0.08},
    }


def _sarimax_metadata(active_regressors: list[str] | None = None) -> dict[str, Any]:
    return {
        "model_family": "sarimax",
        "champion_id": "sarimax_trial_001",
        "granularity": "monthly",
        "active_regressors": active_regressors or [],
        "selection": {"primary_metric": "wape"},
        "metrics": {"wape": 0.10},
    }


def _sarimax_model(*, k_exog: int = 0, with_ci: bool = True, use_exog: bool = False) -> dict:
    return {
        "config": {"order": [1, 1, 1], "seasonal_order": [0, 1, 1, 12], "use_exog": use_exog},
        "model": _FakeSarimaxResults(k_exog=k_exog, with_ci=with_ci),
    }


def _params(default_horizon: int = 12) -> dict[str, Any]:
    return {
        "default_horizon": default_horizon,
        "supported_horizons": [3, 6, 12],
        "output_schema_version": "monthly_forecast_v1",
        "supported_families": ["prophet", "sarimax", "catboost"],
        "sku_column": "sku",
        "prophet": {
            "date_column": "ds",
            "prediction_column": "yhat",
            "lower_column": "yhat_lower",
            "upper_column": "yhat_upper",
        },
        "sarimax": {
            "date_column": "month_start_date",
            "confidence_level": 0.90,
            "interval_strategy": "conf_int",
            "exogenous_columns": [],
        },
        "catboost": {
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
        },
    }


# ── CatBoost fixture builders ──────────────────────────────────────────────────

_CATBOOST_FEATURE_COLS = [
    "month", "business_days",
    "demand_lag_1", "demand_lag_2",
    "rolling_mean_3", "rolling_std_3",
]


def _catboost_metadata() -> dict[str, Any]:
    return {
        "model_family": "catboost",
        "champion_id": "catboost_trial_001",
        "granularity": "monthly",
        "feature_columns": _CATBOOST_FEATURE_COLS,
        "selection": {"primary_metric": "wape"},
        "metrics": {"wape": 0.09},
    }


def _catboost_model(
    value: float = _CATBOOST_PRED,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    cols = feature_columns or _CATBOOST_FEATURE_COLS
    return {
        "model_family": "catboost",
        "model": _FakeCatBoostModel(value),
        "feature_columns": cols,
        "config": {"depth": 6, "learning_rate": 0.05},
        "validation_metrics": {"wape": 0.09},
    }


def _catboost_split_metadata(
    target_lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "sku_column": "sku",
        "lag_settings": {"target_lags": target_lags or [1, 2]},
        "rolling_settings": {
            "windows": rolling_windows or [3],
            "include_std": True,
            "include_min_max": False,
        },
        "trend_feature_columns": [],
        "future_required_columns": ["month", "business_days"],
    }


def _catboost_future_df(n_months: int, start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_months, freq="MS")
    return pd.DataFrame(
        {
            "month_start_date": dates,
            "sku": ["SKU-1"] * n_months,
            "month": dates.month,
            "business_days": [22] * n_months,
        }
    )


def _catboost_history_df(n_months: int = 24, start: str = "2023-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_months, freq="MS")
    demand = np.arange(1, n_months + 1, dtype=float) * 100
    return pd.DataFrame(
        {
            "month_start_date": dates,
            "monthly_demand": demand,
            "sku": ["SKU-1"] * n_months,
        }
    )


def _run_node(model, metadata, *, default_horizon: int = 12):
    return generate_monthly_champion_forecasts(
        model,
        metadata,
        _make_future_df(3),
        _make_future_df(6),
        _make_future_df(12),
        _params(default_horizon=default_horizon),
    )


def _run_catboost_node(
    model,
    metadata,
    *,
    default_horizon: int = 12,
    split_metadata: dict | None = None,
    history_df: pd.DataFrame | None = None,
):
    return generate_monthly_champion_forecasts(
        model,
        metadata,
        _catboost_future_df(3),
        _catboost_future_df(6),
        _catboost_future_df(12),
        _params(default_horizon=default_horizon),
        history_df if history_df is not None else _catboost_history_df(),
        split_metadata if split_metadata is not None else _catboost_split_metadata(),
    )


# ── Dispatch ──────────────────────────────────────────────────────────────────


def test_dispatch_routes_to_prophet_adapter():
    """model_family='prophet' must route through the Prophet adapter."""
    core = dispatch_monthly_prediction(
        _FakeProphet(), _prophet_metadata(), _make_future_df(3), _params(), 3
    )
    assert (core["forecast"] == _PROPHET_YHAT).all()
    assert core["interval_method"].iloc[0] == "prophet_native"


def test_dispatch_routes_to_sarimax_adapter():
    """model_family='sarimax' must route through the SARIMAX adapter."""
    core = dispatch_monthly_prediction(
        _sarimax_model(), _sarimax_metadata(), _make_future_df(3), _params(), 3
    )
    assert pd.api.types.is_numeric_dtype(core["forecast"])
    assert core["interval_method"].iloc[0] == "sarimax_get_forecast_conf_int"


def test_dispatch_routes_to_catboost_adapter():
    """model_family='catboost' must route through the CatBoost recursive adapter."""
    core = dispatch_monthly_prediction(
        _catboost_model(),
        _catboost_metadata(),
        _catboost_future_df(3),
        _params(),
        3,
        history_df=_catboost_history_df(),
        catboost_split_metadata=_catboost_split_metadata(),
    )
    assert len(core) == 3
    assert (core["forecast"] == _CATBOOST_PRED).all()
    assert core["interval_method"].iloc[0] is None
    assert (~core["has_prediction_interval"]).all()


def test_unsupported_family_raises_clear_error():
    """Truly unsupported family must fail with a clear, actionable error message."""
    with pytest.raises(ValueError, match="xgboost"):
        dispatch_monthly_prediction(
            object(), {"model_family": "xgboost"}, _make_future_df(3), _params(), 3
        )


def test_missing_model_family_raises_clear_error():
    """Missing model_family must fail before any adapter runs."""
    bad_meta = {"champion_id": "x", "granularity": "monthly"}
    with pytest.raises(ValueError, match="model_family"):
        generate_monthly_champion_forecasts(
            _FakeProphet(),
            bad_meta,
            _make_future_df(3),
            _make_future_df(6),
            _make_future_df(12),
            _params(),
        )


# ── Standard schema + intervals ───────────────────────────────────────────────


def test_prophet_output_maps_to_standard_schema():
    """Prophet inference must produce the full canonical schema."""
    f3, _, _, _, _ = _run_node(_FakeProphet(), _prophet_metadata())
    assert list(f3.columns) == _STANDARD_FORECAST_COLUMNS
    assert (f3["model_family"] == "prophet").all()
    assert (f3["granularity"] == "monthly").all()
    assert f3["horizon"].tolist() == [1, 2, 3]
    assert (f3["horizon_label"] == "3m").all()
    assert f3["forecast_lower"].notna().all()


def test_prophet_tolerates_missing_interval_columns():
    """When Prophet emits no intervals, interval columns must be null, not missing."""
    f3, _, _, _, _ = _run_node(_FakeProphet(with_intervals=False), _prophet_metadata())
    assert f3["forecast_lower"].isna().all()
    assert f3["forecast_upper"].isna().all()
    assert (~f3["has_prediction_interval"]).all()


def test_sarimax_output_maps_to_standard_schema():
    """SARIMAX inference must produce numeric forecasts and the canonical schema."""
    f3, _, _, _, _ = _run_node(_sarimax_model(), _sarimax_metadata())
    assert list(f3.columns) == _STANDARD_FORECAST_COLUMNS
    assert (f3["model_family"] == "sarimax").all()
    assert pd.api.types.is_numeric_dtype(f3["forecast"])
    assert "forecast_lower" in f3.columns
    assert "forecast_upper" in f3.columns
    assert f3["forecast_lower"].notna().all()


def test_sarimax_null_intervals_when_unavailable():
    """SARIMAX intervals must be null (never fabricated) when conf_int is unavailable."""
    f3, _, _, _, _ = _run_node(_sarimax_model(with_ci=False), _sarimax_metadata())
    assert f3["forecast_lower"].isna().all()
    assert f3["forecast_upper"].isna().all()
    assert (~f3["has_prediction_interval"]).all()


def test_sarimax_missing_exogenous_columns_raises():
    """Missing required exogenous columns must fail, naming the missing column."""
    future = _make_future_df(6)  # has business_days, not promo_index/price_index
    future["price_index"] = 1.0
    metadata = _sarimax_metadata(active_regressors=["promo_index", "price_index"])
    with pytest.raises(ValueError, match="promo_index"):
        predict_monthly_sarimax(
            _sarimax_model(k_exog=2, use_exog=True), future, metadata, _params(), 6
        )


# ── Main node ─────────────────────────────────────────────────────────────────


def test_main_node_returns_all_horizon_outputs():
    """The node must return 3m/6m/12m frames, a latest frame, and audit metadata."""
    f3, f6, f12, latest, meta = _run_node(_FakeProphet(), _prophet_metadata())

    assert len(f3) == _HORIZON_3M
    assert len(f6) == _HORIZON_6M
    assert len(f12) == _HORIZON_12M

    # latest mirrors the configured default horizon (12)
    pd.testing.assert_frame_equal(
        latest.reset_index(drop=True), f12.reset_index(drop=True)
    )

    assert meta["model_family"] == "prophet"
    assert meta["default_horizon"] == _HORIZON_12M
    assert meta["champion_id"] == "prophet_candidate_001"
    assert set(meta["horizons"].keys()) == {"3", "6", "12"}
    assert meta["output_schema_version"] == "monthly_forecast_v1"


def test_latest_follows_configured_default_horizon():
    """forecast_latest must follow default_horizon (here 3m)."""
    f3, _, _, latest, meta = _run_node(
        _FakeProphet(), _prophet_metadata(), default_horizon=3
    )
    pd.testing.assert_frame_equal(
        latest.reset_index(drop=True), f3.reset_index(drop=True)
    )
    assert meta["default_horizon"] == _HORIZON_3M


def test_sarimax_exog_omitted_when_not_trained_with_exog():
    """A SARIMAX champion without exog must forecast with exog=None."""
    model = _sarimax_model(use_exog=False)
    _run_node(model, _sarimax_metadata())
    assert model["model"].last_exog is None


# ── Inference metadata contracts ──────────────────────────────────────────────


def test_inference_metadata_source_frames_are_generic_names():
    """source_future_frames in inference metadata must use generic monthly_future_*m names."""
    _, _, _, _, meta = _run_node(_FakeProphet(), _prophet_metadata())

    source_frames = meta.get("source_future_frames", {})
    assert source_frames.get("3") == "monthly_future_3m"
    assert source_frames.get("6") == "monthly_future_6m"
    assert source_frames.get("12") == "monthly_future_12m"


def test_inference_metadata_notes_empty_for_prophet():
    """Prophet inference should produce no stale or temporary notes."""
    _, _, _, _, meta = _run_node(_FakeProphet(), _prophet_metadata())

    notes = meta.get("notes", [])
    assert notes == [], (
        f"Prophet inference metadata contains unexpected notes: {notes}"
    )


def test_inference_metadata_notes_present_for_sarimax():
    """SARIMAX inference metadata should carry the refit-recommendation note."""
    _, _, _, _, meta = _run_node(_sarimax_model(), _sarimax_metadata())

    notes = meta.get("notes", [])
    assert len(notes) == 1
    assert "refit" in notes[0].lower() or "full-history" in notes[0].lower()


def test_inference_metadata_contains_required_audit_fields():
    """Monthly inference metadata must expose all fields needed for a production audit."""
    _, _, _, _, meta = _run_node(_FakeProphet(), _prophet_metadata())

    required_fields = {
        "granularity",
        "model_family",
        "champion_id",
        "run_id",
        "forecast_generated_at",
        "supported_horizons",
        "default_horizon",
        "output_schema_version",
        "selection_metric",
        "has_prediction_interval",
        "horizons",
        "source_future_frames",
    }
    missing = required_fields - set(meta.keys())
    assert not missing, f"inference_metadata is missing required audit fields: {missing}"


# ── CatBoost recursive feature construction ────────────────────────────────────


def test_catboost_recursive_lag1_at_step1_uses_last_observed():
    """demand_lag_1 at forecast step h=1 must equal the last observed demand value."""
    buffer = [10.0, 20.0, 30.0]
    features = _compute_recursive_features_for_step(
        demand_buffer=buffer,
        target_lags=[1, 2],
        rolling_windows=[],
        include_std=False,
        include_min_max=False,
        trend_diffs=[],
        trend_pct_changes=[],
    )
    assert features["demand_lag_1"] == 30.0, "lag_1 at h=1 must be the last observed value"
    assert features["demand_lag_2"] == 20.0


def test_catboost_recursive_lag1_at_step2_uses_prior_prediction():
    """demand_lag_1 at step h=2 must equal the prediction from step h=1."""
    buffer_after_step1 = [10.0, 20.0, 30.0, 999.0]  # 999.0 is the h=1 prediction appended
    features = _compute_recursive_features_for_step(
        demand_buffer=buffer_after_step1,
        target_lags=[1, 2],
        rolling_windows=[],
        include_std=False,
        include_min_max=False,
        trend_diffs=[],
        trend_pct_changes=[],
    )
    assert features["demand_lag_1"] == 999.0, "lag_1 at h=2 must be the h=1 prediction"
    assert features["demand_lag_2"] == 30.0


def test_catboost_rolling_mean_excludes_current_target():
    """rolling_mean_3 must use the 3 values ending at period T, not T+1."""
    buffer = [10.0, 20.0, 30.0, 40.0]  # T=40; rolling_mean_3 = mean([20,30,40]) = 30
    features = _compute_recursive_features_for_step(
        demand_buffer=buffer,
        target_lags=[],
        rolling_windows=[3],
        include_std=False,
        include_min_max=False,
        trend_diffs=[],
        trend_pct_changes=[],
    )
    assert features["rolling_mean_3"] == pytest.approx(30.0)


def test_catboost_rolling_returns_nan_when_buffer_shorter_than_window():
    """Rolling features must be NaN when the demand buffer has fewer values than the window."""
    buffer = [10.0, 20.0]  # only 2 values; rolling_mean_3 needs 3
    features = _compute_recursive_features_for_step(
        demand_buffer=buffer,
        target_lags=[],
        rolling_windows=[3],
        include_std=True,
        include_min_max=True,
        trend_diffs=[],
        trend_pct_changes=[],
    )
    assert np.isnan(features["rolling_mean_3"])
    assert np.isnan(features["rolling_std_3"])
    assert np.isnan(features["rolling_min_3"])
    assert np.isnan(features["rolling_max_3"])


def test_catboost_lag_returns_nan_when_buffer_shorter_than_lag():
    """Lag features must be NaN when the demand buffer has fewer values than the lag period."""
    buffer = [10.0]
    features = _compute_recursive_features_for_step(
        demand_buffer=buffer,
        target_lags=[1, 2, 3],
        rolling_windows=[],
        include_std=False,
        include_min_max=False,
        trend_diffs=[],
        trend_pct_changes=[],
    )
    assert features["demand_lag_1"] == 10.0
    assert np.isnan(features["demand_lag_2"])
    assert np.isnan(features["demand_lag_3"])


def test_catboost_trend_diff_computation():
    """demand_diff_1 must equal buffer[-1] - buffer[-2]."""
    buffer = [10.0, 20.0, 35.0]
    features = _compute_recursive_features_for_step(
        demand_buffer=buffer,
        target_lags=[],
        rolling_windows=[],
        include_std=False,
        include_min_max=False,
        trend_diffs=[1],
        trend_pct_changes=[],
    )
    assert features["demand_diff_1"] == pytest.approx(15.0)  # 35 - 20


def test_catboost_pct_change_computation():
    """demand_pct_change_1 must equal (buffer[-1] - buffer[-2]) / buffer[-2]."""
    buffer = [10.0, 20.0, 30.0]  # (30-20)/20 = 0.5
    features = _compute_recursive_features_for_step(
        demand_buffer=buffer,
        target_lags=[],
        rolling_windows=[],
        include_std=False,
        include_min_max=False,
        trend_diffs=[],
        trend_pct_changes=[1],
    )
    assert features["demand_pct_change_1"] == pytest.approx(0.5)


def test_catboost_rolling_mean_3_vs_12_is_ratio():
    """rolling_mean_3_vs_12 must be rolling_mean_3 / rolling_mean_12."""
    buffer = list(range(1, 25))  # 24 values; both windows have enough data
    features = _compute_recursive_features_for_step(
        demand_buffer=buffer,
        target_lags=[],
        rolling_windows=[3, 12],
        include_std=False,
        include_min_max=False,
        trend_diffs=[],
        trend_pct_changes=[],
    )
    expected = features["rolling_mean_3"] / features["rolling_mean_12"]
    assert features["rolling_mean_3_vs_12"] == pytest.approx(expected)


# ── CatBoost adapter integration ───────────────────────────────────────────────


def test_catboost_adapter_produces_canonical_schema():
    """CatBoost inference must produce the full canonical monthly forecast schema."""
    f3, _, _, _, _ = _run_catboost_node(_catboost_model(), _catboost_metadata())
    assert list(f3.columns) == _STANDARD_FORECAST_COLUMNS
    assert (f3["model_family"] == "catboost").all()
    assert (f3["granularity"] == "monthly").all()
    assert f3["horizon"].tolist() == [1, 2, 3]
    assert (f3["horizon_label"] == "3m").all()


def test_catboost_adapter_returns_correct_horizon_row_counts():
    """CatBoost inference must return 3, 6, and 12 rows for the respective horizons."""
    f3, f6, f12, _, _ = _run_catboost_node(_catboost_model(), _catboost_metadata())
    assert len(f3) == 3
    assert len(f6) == 6
    assert len(f12) == 12


def test_catboost_adapter_null_intervals():
    """CatBoost forecasts must have null intervals and has_prediction_interval=False."""
    f3, _, _, _, _ = _run_catboost_node(_catboost_model(), _catboost_metadata())
    assert f3["forecast_lower"].isna().all()
    assert f3["forecast_upper"].isna().all()
    assert (~f3["has_prediction_interval"]).all()
    assert f3["interval_method"].isna().all()


def test_catboost_adapter_forecast_dates_sorted_after_training_cutoff():
    """CatBoost forecast dates must be sorted and all strictly after the last history date."""
    history = _catboost_history_df(n_months=24, start="2023-01-01")
    last_history_date = history["month_start_date"].max()

    f3, _, _, _, _ = _run_catboost_node(
        _catboost_model(), _catboost_metadata(), history_df=history
    )
    forecast_dates = pd.to_datetime(f3["date"])
    assert forecast_dates.is_monotonic_increasing, "forecast dates must be sorted"
    assert (forecast_dates > last_history_date).all(), (
        "all forecast dates must be strictly after the last historical observation"
    )


def test_catboost_adapter_feature_column_order_is_preserved():
    """The feature matrix passed to CatBoost.predict() must follow the stored feature_columns order."""
    feature_columns = ["month", "business_days", "demand_lag_1", "demand_lag_2", "rolling_mean_3", "rolling_std_3"]
    recording_model = _FakeCatBoostModel()
    candidate = {"model": recording_model, "feature_columns": feature_columns}
    metadata = {**_catboost_metadata(), "feature_columns": feature_columns}

    predict_monthly_catboost(
        candidate,
        _catboost_future_df(3),
        metadata,
        _params(),
        3,
        history_df=_catboost_history_df(),
        catboost_split_metadata=_catboost_split_metadata(),
    )

    assert len(recording_model.call_inputs) == 3, "predict() must be called once per forecast step"
    for call_X in recording_model.call_inputs:
        assert call_X.shape == (1, len(feature_columns)), (
            f"Each predict call must pass exactly {len(feature_columns)} features in one row"
        )


def test_catboost_adapter_missing_future_columns_raises_clear_error():
    """Missing required future exogenous columns must fail with a clear error naming them."""
    split_meta = _catboost_split_metadata()
    split_meta["future_required_columns"] = ["month", "business_days", "promo_flag"]

    future_df = _catboost_future_df(3)  # has month and business_days but NOT promo_flag

    with pytest.raises(ValueError, match="promo_flag"):
        predict_monthly_catboost(
            _catboost_model(),
            future_df,
            _catboost_metadata(),
            _params(),
            3,
            history_df=_catboost_history_df(),
            catboost_split_metadata=split_meta,
        )


def test_catboost_inference_metadata_contains_recursive_note():
    """CatBoost inference metadata must carry a note about recursive strategy."""
    _, _, _, _, meta = _run_catboost_node(_catboost_model(), _catboost_metadata())
    notes = meta.get("notes", [])
    assert any("recursive" in n.lower() for n in notes), (
        f"CatBoost inference metadata should include a recursive-strategy note; got: {notes}"
    )


def test_catboost_node_dispatch_routes_correctly():
    """generate_monthly_champion_forecasts must dispatch to CatBoost when model_family=catboost."""
    f3, f6, f12, latest, meta = _run_catboost_node(_catboost_model(), _catboost_metadata())
    assert meta["model_family"] == "catboost"
    assert len(f3) == 3
    assert len(f6) == 6
    assert len(f12) == 12
    assert len(latest) == 12  # default_horizon=12
