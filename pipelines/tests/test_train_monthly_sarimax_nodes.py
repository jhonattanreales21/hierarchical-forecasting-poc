"""Tests for monthly SARIMAX training nodes (Optuna-based tuning)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hdf_pipelines.pipelines.train_monthly.sarimax.nodes import (
    _build_sarimax_config_from_trial,
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
    use_exog: bool = False,
    top_n: int = 2,
    max_trials: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Build Optuna-based SARIMAX params for testing with a small search budget."""
    return {
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "objective": {"metric": "wape", "direction": "minimize"},
        "top_n_prechampions": top_n,
        "metrics": {"mase_seasonal_period": 12, "epsilon": 1.0},
        "tuning": {
            "optimizer": "optuna",
            "max_trials": max_trials,
            "use_exog": use_exog,
            "sampler": {"name": "tpe", "seed": seed},
            "search_space": {
                "p": {"type": "int", "low": 0, "high": 1},
                "d": {"type": "int", "low": 0, "high": 1},
                "q": {"type": "int", "low": 0, "high": 1},
                "P": {"type": "int", "low": 0, "high": 1},
                "D": {"type": "int", "low": 0, "high": 1},
                "Q": {"type": "int", "low": 0, "high": 1},
                "trend": {"type": "categorical", "choices": ["none", "n"]},
            },
            "fixed_params": {
                "s": 12,
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
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


class TestBuildSarimaxConfigFromTrial:
    def test_assembles_order_and_seasonal_order(self):
        params = {
            "p": 1, "d": 1, "q": 1,
            "P": 0, "D": 1, "Q": 1,
            "trend": "none",
            "enforce_stationarity": False,
            "enforce_invertibility": False,
            "s": 12,
        }
        config = _build_sarimax_config_from_trial(params, s=12)
        assert config["order"] == [1, 1, 1]
        assert config["seasonal_order"] == [0, 1, 1, 12]

    def test_trend_none_string_maps_to_python_none(self):
        params = {"p": 0, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1, "trend": "none"}
        config = _build_sarimax_config_from_trial(params, s=12)
        assert config["trend"] is None

    def test_trend_string_values_preserved(self):
        for trend_str in ["n", "c", "t", "ct"]:
            params = {"p": 0, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1, "trend": trend_str}
            config = _build_sarimax_config_from_trial(params, s=12)
            assert config["trend"] == trend_str

    def test_s_sets_seasonal_period(self):
        params = {"p": 1, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1, "trend": "none"}
        config = _build_sarimax_config_from_trial(params, s=6)
        assert config["seasonal_order"][3] == 6

    def test_enforce_flags_default_to_false(self):
        params = {"p": 0, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1, "trend": "none"}
        config = _build_sarimax_config_from_trial(params, s=12)
        assert config["enforce_stationarity"] is False
        assert config["enforce_invertibility"] is False


class TestComputeValidationMetrics:
    def test_returns_all_required_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        y_train = np.linspace(1.0, 10.0, 20)
        metrics = _compute_validation_metrics(y_true, y_pred, y_train, 12, 1.0)
        assert set(metrics.keys()) >= {"wape", "mase", "rmse", "bias", "mae"}

    def test_mase_uses_configured_seasonal_period(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        y_train = np.linspace(1.0, 10.0, 24)
        metrics_12 = _compute_validation_metrics(y_true, y_pred, y_train, 12, 1.0)
        metrics_3 = _compute_validation_metrics(y_true, y_pred, y_train, 3, 1.0)
        assert metrics_12["mase"] is not None
        assert metrics_3["mase"] is not None

    def test_mase_is_none_when_train_too_short(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([10.0, 20.0])
        y_train = np.array([5.0, 6.0, 7.0])
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
        assert successful.iloc[0]["trial_id"] == "t2"
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
        assert successful.iloc[0]["trial_id"] == "t1"


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

    def test_training_metadata_contains_optuna_fields(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params()
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            _, _, _, _, training_metadata, _ = train_and_evaluate_monthly_sarimax_candidates(
                train, val, metadata, params
            )

        assert training_metadata["optimizer"] == "optuna"
        assert "n_trials_completed" in training_metadata
        assert "n_trials_pruned" in training_metadata
        assert "study_direction" in training_metadata
        assert training_metadata["study_direction"] == "minimize"

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
        params = _build_params(max_trials=3, top_n=2)
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
        params = _build_params(max_trials=3)

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
        params["metrics"]["mase_seasonal_period"] = 6
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            _, val_metrics_df, _, _, training_metadata, _ = (
                train_and_evaluate_monthly_sarimax_candidates(train, val, metadata, params)
            )

        assert training_metadata["seasonal_period"] == 6
        assert (val_metrics_df["seasonal_period"] == 6).all()

    def test_ranks_successful_candidates_deterministically(self):
        """With a fixed TPE seed, two identical runs must produce the same trial order."""
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params(max_trials=3, seed=42)
        FakeSARIMAX = _make_fake_sarimax_class()

        results_a: list[str] = []
        results_b: list[str] = []
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

    def test_candidate_models_contain_exogenous_columns_when_exog_used(self):
        train, val = _build_train_val_inputs()
        train["some_exog"] = np.linspace(0.1, 1.0, len(train))
        val["some_exog"] = np.linspace(0.1, 1.0, len(val))
        metadata = _build_split_metadata(exog_cols=["some_exog"])
        params = _build_params(use_exog=True)
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            _, _, _, candidate_models, _, _ = train_and_evaluate_monthly_sarimax_candidates(
                train, val, metadata, params
            )

        for _cid, entry in candidate_models.items():
            assert "exogenous_columns" in entry
            assert entry["exogenous_columns"] == ["some_exog"]

    def test_candidate_monthly_sarimax_contains_exogenous_columns(self):
        train, val = _build_train_val_inputs()
        train["feat_a"] = np.linspace(0.0, 1.0, len(train))
        val["feat_a"] = np.linspace(0.0, 1.0, len(val))
        metadata = _build_split_metadata(exog_cols=["feat_a"])
        params = _build_params(use_exog=True)
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            _, _, _, _, _, candidate = train_and_evaluate_monthly_sarimax_candidates(
                train, val, metadata, params
            )

        assert "exogenous_columns" in candidate
        assert candidate["exogenous_columns"] == ["feat_a"]

    def test_prechampion_configs_candidates_contain_exogenous_columns(self):
        train, val = _build_train_val_inputs()
        train["exog_feature"] = np.linspace(0.0, 1.0, len(train))
        val["exog_feature"] = np.linspace(0.0, 1.0, len(val))
        metadata = _build_split_metadata(exog_cols=["exog_feature"])
        params = _build_params(use_exog=True)
        FakeSARIMAX = _make_fake_sarimax_class()

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FakeSARIMAX,
        ):
            _, _, prechampion_configs, _, _, _ = train_and_evaluate_monthly_sarimax_candidates(
                train, val, metadata, params
            )

        for cand in prechampion_configs["candidates"]:
            assert "exogenous_columns" in cand
            assert cand["exogenous_columns"] == ["exog_feature"]

    def test_max_failed_trials_stops_optuna_search_early(self):
        """max_failed_trials callback should stop study before all max_trials run."""
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params(max_trials=5)
        params["max_failed_trials"] = 1

        call_count: dict[str, int] = {"n": 0}

        class FirstSucceedsThenFails:
            def __init__(self, endog, exog=None, **kwargs):
                call_count["n"] += 1
                # Trial 1 succeeds; everything after fails.
                if call_count["n"] > 1:
                    raise ValueError("synthetic failure for early-stopping test")

            def fit(self, disp=False):
                result = MagicMock()
                result.get_forecast.return_value.predicted_mean = pd.Series([15.5, 16.5, 17.5])
                return result

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=FirstSucceedsThenFails,
        ):
            tuning_df, _, _, _, training_metadata, _ = (
                train_and_evaluate_monthly_sarimax_candidates(train, val, metadata, params)
            )

        # Study stopped early: fewer rows than max_trials=5.
        assert len(tuning_df) < 5, f"Expected early stopping; got {len(tuning_df)} trial rows"
        assert training_metadata["n_trials_failed"] >= 1
        assert training_metadata["n_trials_successful"] >= 1

    def test_nan_predictions_treated_as_failed_trial(self):
        train, val = _build_train_val_inputs()
        metadata = _build_split_metadata()
        params = _build_params()

        class NanPredictingSARIMAX:
            def __init__(self, endog, exog=None, **kwargs):
                pass

            def fit(self, disp=False):
                result = MagicMock()
                result.get_forecast.return_value.predicted_mean = pd.Series(
                    [float("nan"), float("nan"), float("nan")]
                )
                return result

        with patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=NanPredictingSARIMAX,
        ):
            with pytest.raises(RuntimeError, match="All SARIMAX trials failed"):
                train_and_evaluate_monthly_sarimax_candidates(train, val, metadata, params)
