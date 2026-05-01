"""Tests for monthly Prophet forecast inference nodes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from hdf_pipelines.pipelines.forecast_inference.nodes import (
    generate_monthly_prophet_forecasts,
)

_HORIZON_3M = 3
_HORIZON_6M = 6
_HORIZON_12M = 12


def _make_future_df(n_months: int, start: str = "2025-01-01") -> pd.DataFrame:
    """Build a minimal future regressor DataFrame for Prophet inference."""
    return pd.DataFrame(
        {
            "ds": pd.date_range(start, periods=n_months, freq="MS"),
            "sku": ["SKU-1"] * n_months,
            "business_days": [22] * n_months,
        }
    )


def _make_champion_metadata() -> dict[str, Any]:
    return {
        "champion_id": "prophet_candidate_001",
        "model_family": "prophet",
        "granularity": "monthly",
        "active_regressors": ["business_days"],
        "selection_metric": "mape",
        "selection_metric_value": 0.08,
        "business_success_flag": True,
    }


def _make_params(latest_horizon: int = 12) -> dict[str, Any]:
    return {
        "date_column": "ds",
        "sku_column": "sku",
        "output": {
            "include_prediction_intervals": True,
            "latest_output_horizon_months": latest_horizon,
            "model_family": "prophet",
            "model_granularity": "monthly",
        },
        "validation": {
            "fail_on_missing_regressors": True,
            "fail_on_null_regressors": True,
            "fail_if_future_contains_y": True,
        },
    }


def _make_mock_prophet(active_regressors: list[str]) -> MagicMock:
    """Return a mock Prophet model whose predict() echoes back ds with deterministic yhat."""

    def _predict(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ds": df["ds"].values,
                "yhat": [100.0] * len(df),
                "yhat_lower": [90.0] * len(df),
                "yhat_upper": [110.0] * len(df),
            }
        )

    mock = MagicMock()
    mock.predict.side_effect = _predict
    return mock


class TestGenerateMonthlyProphetForecasts:
    """Covers forecast schema, horizon lengths, and forecast_latest aliasing."""

    def test_output_schema_contains_required_columns(self):
        """Every horizon output must carry all columns from _FORECAST_COLUMN_ORDER."""
        meta = _make_champion_metadata()
        model = _make_mock_prophet(meta["active_regressors"])
        params = _make_params(latest_horizon=12)

        forecast_3m, forecast_6m, forecast_12m, forecast_latest, inf_meta = (
            generate_monthly_prophet_forecasts(
                model,
                meta,
                _make_future_df(3),
                _make_future_df(6),
                _make_future_df(12),
                params,
            )
        )

        required_cols = {
            "ds",
            "sku",
            "horizon_month",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "model_family",
            "model_granularity",
            "champion_id",
            "forecast_created_at",
            "forecast_horizon_months",
        }
        for df, label in [
            (forecast_3m, "3m"),
            (forecast_6m, "6m"),
            (forecast_12m, "12m"),
            (forecast_latest, "latest"),
        ]:
            missing = required_cols - set(df.columns)
            assert not missing, f"forecast_{label} is missing columns: {missing}"

    def test_horizon_row_counts_are_exact(self):
        """Each horizon output must contain exactly the configured number of rows."""
        meta = _make_champion_metadata()
        model = _make_mock_prophet(meta["active_regressors"])

        forecast_3m, forecast_6m, forecast_12m, _, _ = (
            generate_monthly_prophet_forecasts(
                model,
                meta,
                _make_future_df(3),
                _make_future_df(6),
                _make_future_df(12),
                _make_params(),
            )
        )

        assert len(forecast_3m) == _HORIZON_3M
        assert len(forecast_6m) == _HORIZON_6M
        assert len(forecast_12m) == _HORIZON_12M

    def test_horizon_month_starts_at_1_and_is_ordered(self):
        """horizon_month must be 1-based and strictly increasing for each horizon."""
        meta = _make_champion_metadata()
        model = _make_mock_prophet(meta["active_regressors"])

        forecast_3m, _, forecast_12m, _, _ = generate_monthly_prophet_forecasts(
            model,
            meta,
            _make_future_df(3),
            _make_future_df(6),
            _make_future_df(12),
            _make_params(),
        )

        assert forecast_3m["horizon_month"].tolist() == list(range(1, _HORIZON_3M + 1))
        assert forecast_12m["horizon_month"].tolist() == list(
            range(1, _HORIZON_12M + 1)
        )

    def test_forecast_latest_matches_configured_latest_horizon(self):
        """forecast_latest must be an exact copy of the forecast for latest_output_horizon_months."""
        meta = _make_champion_metadata()
        model = _make_mock_prophet(meta["active_regressors"])

        _, _, forecast_12m, forecast_latest, _ = generate_monthly_prophet_forecasts(
            model,
            meta,
            _make_future_df(3),
            _make_future_df(6),
            _make_future_df(12),
            _make_params(latest_horizon=12),
        )

        pd.testing.assert_frame_equal(
            forecast_latest.reset_index(drop=True),
            forecast_12m.reset_index(drop=True),
        )
        assert (forecast_latest["forecast_horizon_months"] == _HORIZON_12M).all()

    def test_forecast_horizon_months_column_matches_horizon(self):
        """The forecast_horizon_months column must equal the horizon for each output table."""
        meta = _make_champion_metadata()
        model = _make_mock_prophet(meta["active_regressors"])

        forecast_3m, forecast_6m, forecast_12m, _, _ = (
            generate_monthly_prophet_forecasts(
                model,
                meta,
                _make_future_df(3),
                _make_future_df(6),
                _make_future_df(12),
                _make_params(),
            )
        )

        assert (forecast_3m["forecast_horizon_months"] == _HORIZON_3M).all()
        assert (forecast_6m["forecast_horizon_months"] == _HORIZON_6M).all()
        assert (forecast_12m["forecast_horizon_months"] == _HORIZON_12M).all()

    def test_ds_are_month_start_dates(self):
        """All ds values in forecast outputs must be first-of-month dates."""
        meta = _make_champion_metadata()
        model = _make_mock_prophet(meta["active_regressors"])

        _, _, forecast_12m, _, _ = generate_monthly_prophet_forecasts(
            model,
            meta,
            _make_future_df(3),
            _make_future_df(6),
            _make_future_df(12),
            _make_params(),
        )

        dates = pd.to_datetime(forecast_12m["ds"])
        assert (dates.dt.day == 1).all(), "All ds values must be month-start (day=1)."

    def test_invalid_latest_horizon_raises(self):
        """A latest_output_horizon_months not in [3, 6, 12] must raise ValueError."""
        meta = _make_champion_metadata()
        model = _make_mock_prophet(meta["active_regressors"])

        with pytest.raises(ValueError, match="latest_output_horizon_months"):
            generate_monthly_prophet_forecasts(
                model,
                meta,
                _make_future_df(3),
                _make_future_df(6),
                _make_future_df(12),
                _make_params(
                    latest_horizon=24
                ),  # noqa: PLR2004  # not a generated horizon
            )

    def test_future_dataset_with_target_column_raises(self):
        """Future datasets that accidentally contain 'y' must be rejected."""
        meta = _make_champion_metadata()
        model = _make_mock_prophet(meta["active_regressors"])
        bad_future = _make_future_df(3)
        bad_future["y"] = 0.0  # target leakage

        with pytest.raises(ValueError, match="column 'y'"):
            generate_monthly_prophet_forecasts(
                model,
                meta,
                bad_future,
                _make_future_df(6),
                _make_future_df(12),
                _make_params(),
            )
