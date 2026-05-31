"""Tests for metadata-driven monthly champion forecast inference (Phase 6).

Covers:
- metadata-driven dispatch to the Prophet and SARIMAX adapters;
- unsupported family and missing-metadata failures;
- the canonical standard forecast schema and nullable interval handling;
- SARIMAX exogenous validation;
- the main node's horizon outputs, latest aliasing, and audit metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from hdf_pipelines.pipelines.forecast_inference.adapters import (
    dispatch_monthly_prediction,
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
        "supported_families": ["prophet", "sarimax"],
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
    }


def _run_node(model, metadata, *, default_horizon: int = 12):
    return generate_monthly_champion_forecasts(
        model,
        metadata,
        _make_future_df(3),
        _make_future_df(6),
        _make_future_df(12),
        _params(default_horizon=default_horizon),
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


def test_unsupported_family_raises_clear_error():
    """CatBoost must be rejected with an actionable Phase 6 message."""
    with pytest.raises(ValueError, match="catboost"):
        dispatch_monthly_prediction(
            object(), {"model_family": "catboost"}, _make_future_df(3), _params(), 3
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
