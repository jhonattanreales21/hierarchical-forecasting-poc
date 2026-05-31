"""Tests for monthly SARIMAX training nodes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hdf_pipelines.pipelines.train_monthly.sarimax.nodes import (
    _build_param_grid,
    _compute_validation_metrics,
    _rank_candidates,
    train_and_evaluate_monthly_sarimax_candidates,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


def _build_train_val_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "month_start_date": pd.date_range("2021-01-01", periods=24, freq="MS"),
            "monthly_demand": np.linspace(5.0, 15.0, 24),
        }
    )
    val = pd.DataFrame(
        {
            "month_start_date": pd.date_range("2023-01-01", periods=3, freq="MS"),
            "monthly_demand": [15.0, 16.0, 17.0],
        }
    )
    return train, val


def _build_split_metadata(exog_cols: list[str] | None = None) -> dict:
    return {
        "model_family": "sarimax",
        "granularity": "monthly",
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "exogenous_columns": exog_cols or [],
    }


def _build_params(
    order_grid: list | None = None,
    seasonal_order_grid: list | None = None,
    use_exog: bool = False,
    top_n: int = 2,
) -> dict[str, Any]:
    return {
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "objective": {"metric": "wape", "direction": "minimize"},
        "top_n_prechampions": top_n,
        "metrics": {"mase_seasonal_period": 12, "epsilon": 1.0},
        "tuning": {
            "order_grid": order_grid or [[1, 1, 1]],
            "seasonal_order_grid": seasonal_order_grid or [[0, 1, 1, 12]],
            "trend_options": [None],
            "enforce_stationarity_options": [False],
            "enforce_invertibility_options": [False],
            "use_exog": use_exog,
        },
    }


def _make_fake_sarimax_class(predictions: list[float] | None = None, fail: bool = False):
    """Return a SARIMAX class mock that produces deterministic predictions."""
    preds = predictions or [15.5, 16.5, 17.5]

    class FakeResult:
        def get_forecast(self, steps: int, exog=None):
            result = MagicMock()
            result.predicted_mean = pd.Series(preds[:steps])
            return result

    class FakeModel:
        def __init__(self, endog, exog=None, **kwargs):
            if fail:
                raise ValueError("synthetic SARIMAX fit failure")

        def fit(self, disp=False):
            return FakeResult()

    return FakeModel


# ── unit tests ────────────────────────────────────────────────────────────────


class TestBuildParamGrid:
    def test_cartesian_product_of_order_and_seasonal(self):
        cfg = {
            "order_grid": [[1, 1, 1], [0, 1, 1]],
            "seasonal_order_grid": [[0, 1, 1, 12], [1, 1, 0, 12]],
            "trend_options": [None],
            "enforce_stationarity_options": [False],
            "enforce_invertibility_options": [False],
        }
        grid = _build_param_grid(cfg)
        assert len(grid) == 4  # 2 × 2 × 1 × 1 × 1

    def test_single_config_produces_one_entry(self):
        cfg = {
            "order_grid": [[1, 1, 1]],
            "seasonal_order_grid": [[0, 1, 1, 12]],
            "trend_options": [None],
            "enforce_stationarity_options": [False],
            "enforce_invertibility_options": [False],
        }
        grid = _build_param_grid(cfg)
        assert len(grid) == 1
        assert grid[0]["order"] == [1, 1, 1]
        assert grid[0]["seasonal_order"] == [0, 1, 1, 12]


class TestComputeValidationMetrics:
    def test_returns_all_required_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        y_train = np.linspace(1.0, 10.0, 20)
        metrics = _compute_validation_metrics(y_true, y_pred, y_train, 12, 1.0)
        assert set(metrics.keys()) >= {"wape", "mase", "rmse", "bias", "mae"}

    def test_mase_uses_configured_seasonal_period(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])  # perfect forecast → mase=0
        y_train = np.linspace(1.0, 10.0, 24)
        metrics_12 = _compute_validation_metrics(y_true, y_pred, y_train, 12, 1.0)
        metrics_3 = _compute_validation_metrics(y_true, y_pred, y_train, 3, 1.0)
        # Both should compute mase (not None) for 24 training points
        assert metrics_12["mase"] is not None
        assert metrics_3["mase"] is not None

    def test_mase_is_none_when_train_too_short(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([10.0, 20.0])
        y_train = np.array([5.0, 6.0, 7.0])  # shorter than seasonal period
        metrics = _compute_validation_metrics(y_true, y_pred, y_train, 12, 1.0)
        assert metrics["mase"] is None

    def test_wape_is_zero_for_perfect_forecast(self):
        y_true = np.array([10.0, 20.0, 30.0])
        metrics = _compute_validation_metrics(y_true, y_true.copy(), y_true, 3, 1.0)
        assert metrics["wape"] == pytest.approx(0.0, abs=1e-9)


class TestRankCandidates:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trial_id": ["t1", "t2", "t3", "t4"],
                "status": ["success", "success", "failed", "success"],
                "objective_value": [0.3, 0.1, None, 0.2],
            }
        )

    def test_success_rows_ranked_ascending_for_minimize(self):
        df = self._make_df()
        ranked = _rank_candidates(df, "wape", "minimize")
        successful = ranked[ranked["status"] == "success"].sort_values("rank")
        assert successful.iloc[0]["trial_id"] == "t2"  # wape=0.1 is best
        assert successful.iloc[1]["trial_id"] == "t4"
        assert successful.iloc[2]["trial_id"] == "t1"

    def test_failed_rows_have_null_rank(self):
        df = self._make_df()
        ranked = _rank_candidates(df, "wape", "minimize")
        failed_row = ranked[ranked["trial_id"] == "t3"]
        assert failed_row["rank"].isna().all()

    def test_successful_rows_ranked_descending_for_maximize(self):
        df = self._make_df()
        ranked = _rank_candidates(df, "wape", "maximize")
        successful = ranked[ranked["status"] == "success"].sort_values("rank")
        assert successful.iloc[0]["trial_id"] == "t1"  # 0.3 is highest


class TestTrainAndEvaluateMonthlySarimaxCandidates:
    def test_emits_required_artifacts(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params()
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            result = train_and_evaluate_monthly_sarimax_candidates(
                train, val, metadata, params
            )

        assert len(result) == 6
        tuning_df, val_metrics_df, prechampion_configs, candidate_models, training_metadata, candidate = result
        assert isinstance(tuning_df, pd.DataFrame)
        assert isinstance(val_metrics_df, pd.DataFrame)
        assert isinstance(prechampion_configs, dict)
        assert isinstance(candidate_models, dict)
        assert isinstance(training_metadata, dict)
        assert isinstance(candidate, dict)

    def test_tuning_df_has_required_columns(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params()
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            tuning_df, *_ = train_and_evaluate_monthly_sarimax_candidates(
                train, val, metadata, params
            )

        required = {"trial_id", "status", "model_family", "granularity", "wape", "rmse", "bias"}
        assert required.issubset(set(tuning_df.columns))

    def test_validation_metrics_include_required_metrics(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params()
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            _, val_metrics_df, *_ = train_and_evaluate_monthly_sarimax_candidates(
                train, val, metadata, params
            )

        assert {"wape", "mase", "rmse", "bias", "mae"}.issubset(set(val_metrics_df.columns))

    def test_candidate_monthly_sarimax_is_rank_one(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params(
            order_grid=[[1, 1, 1], [0, 1, 1]],
            seasonal_order_grid=[[0, 1, 1, 12]],
        )
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            tuning_df, _, _, _, _, candidate = train_and_evaluate_monthly_sarimax_candidates(
                train, val, metadata, params
            )

        assert candidate["rank"] == 1
        assert candidate["model_family"] == "sarimax"
        assert candidate["granularity"] == "monthly"
        rank1_id = tuning_df[tuning_df["rank"] == 1]["trial_id"].iloc[0]
        assert candidate["candidate_id"] == rank1_id

    def test_handles_failed_trial_without_failing_all(self):
        """One failing config should not abort the entire training run."""
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        # Two configs: one will fail, one will succeed via alternating mock
        params = _build_params(
            order_grid=[[1, 1, 1], [2, 1, 1]],
            seasonal_order_grid=[[0, 1, 1, 12]],
        )

        call_count = {"n": 0}

        class PartiallyFailingSARIMAX:
            def __init__(self, endog, exog=None, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise ValueError("first trial synthetic failure")

            def fit(self, disp=False):
                result = MagicMock()
                result.get_forecast.return_value.predicted_mean = pd.Series(
                    [15.5, 16.5, 17.5]
                )
                return result

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=PartiallyFailingSARIMAX,
        ):
            tuning_df, _, _, _, training_metadata, candidate = (
                train_and_evaluate_monthly_sarimax_candidates(train, val, metadata, params)
            )

        failed_rows = tuning_df[tuning_df["status"] == "failed"]
        success_rows = tuning_df[tuning_df["status"] == "success"]
        assert len(failed_rows) >= 1
        assert len(success_rows) >= 1
        assert training_metadata["n_trials_failed"] >= 1
        assert training_metadata["n_trials_successful"] >= 1
        assert candidate["rank"] == 1

    def test_raises_when_all_trials_fail(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params()
        FakeSARIMAX = _make_fake_sarimax_class(fail=True)

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            with pytest.raises(RuntimeError, match="All SARIMAX trials failed"):
                train_and_evaluate_monthly_sarimax_candidates(train, val, metadata, params)

    def test_training_uses_configured_mase_seasonal_period(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params()
        params["metrics"]["mase_seasonal_period"] = 6  # override to 6
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            _, val_metrics_df, _, _, training_metadata, _ = (
                train_and_evaluate_monthly_sarimax_candidates(train, val, metadata, params)
            )

        # Metadata must reflect the configured seasonal period
        assert training_metadata["seasonal_period"] == 6
        # val_metrics_df must record it too
        assert (val_metrics_df["seasonal_period"] == 6).all()

    def test_ranks_successful_candidates_deterministically(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params(
            order_grid=[[1, 1, 1], [0, 1, 1]],
            seasonal_order_grid=[[0, 1, 1, 12]],
            top_n=2,
        )
        FakeSARIMAX = _make_fake_sarimax_class()

        results_a = []
        results_b = []
        for result_list in (results_a, results_b):
            with patch(
                "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
                new=FakeSARIMAX,
            ):
                tuning_df, *_ = train_and_evaluate_monthly_sarimax_candidates(
                    train, val, metadata, params
                )
            result_list.extend(
                tuning_df[tuning_df["status"] == "success"]["trial_id"].tolist()
            )

        assert results_a == results_b

    def test_prechampion_configs_structure(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params()
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            _, _, prechampion_configs, *_ = train_and_evaluate_monthly_sarimax_candidates(
                train, val, metadata, params
            )

        assert prechampion_configs["model_family"] == "sarimax"
        assert prechampion_configs["granularity"] == "monthly"
        assert prechampion_configs["selection_stage"] == "validation"
        assert "candidates" in prechampion_configs
        for cand in prechampion_configs["candidates"]:
            assert "trial_id" in cand
            assert "order" in cand
            assert "seasonal_order" in cand
            assert "metrics" in cand

    def test_raises_on_empty_train(self):
        _, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params()
        empty_train = pd.DataFrame(columns=["month_start_date", "monthly_demand"])
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            with pytest.raises(ValueError, match="empty"):
                train_and_evaluate_monthly_sarimax_candidates(
                    empty_train, val, metadata, params
                )

    def test_raises_on_missing_target_column(self):
        train, val = _build_train_val_inputs()
        bad_train = train.rename(columns={"monthly_demand": "wrong_col"})
        metadata = _build_split_metadata()
        params = _build_params()
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            with pytest.raises(ValueError, match="monthly_demand"):
                train_and_evaluate_monthly_sarimax_candidates(
                    bad_train, val, metadata, params
                )
