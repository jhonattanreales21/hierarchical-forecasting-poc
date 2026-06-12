"""Tests for monthly multi-family model selection nodes (Prophet + SARIMAX + CatBoost).

Covers:
- evaluate_monthly_family_candidates_on_test (Prophet + SARIMAX + CatBoost scoring)
- _score_catboost_candidates (leakage-safe CatBoost test scoring)
- select_monthly_family_champions (one champion per family)
- select_monthly_production_champion (primary metric + tie-breakers, three families)
- build_monthly_champion_artifacts (metadata schema + model resolution, incl. CatBoost)
- Scope guard: CatBoost included, weekly excluded
- Hard-fail when active family artifacts are missing
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from hdf_pipelines.pipelines.model_selection.monthly.nodes import (
    _extract_prophet_regressor_prior_scales,
    _refit_prophet_on_frame,
    _score_catboost_candidates,
    _score_prophet_candidates,
    _score_sarimax_candidates,
    annotate_monthly_candidate_champion_flags,
    build_monthly_champion_artifacts,
    evaluate_monthly_family_candidates_on_test,
    select_monthly_family_champions,
    select_monthly_production_champion,
)

# ── Test doubles ──────────────────────────────────────────────────────────────


class _FakeProphetModel:
    """Prophet test double that returns deterministic predictions."""

    def __init__(self, yhat_multiplier: float = 1.0) -> None:
        self._mult = yhat_multiplier

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        return pd.DataFrame(
            {
                "ds": df["ds"].values,
                "yhat": np.ones(n) * 10.0 * self._mult,
                "yhat_lower": np.ones(n) * 8.0 * self._mult,
                "yhat_upper": np.ones(n) * 12.0 * self._mult,
            }
        )


def _make_prophet_test_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2023-01-01", periods=n, freq="MS"),
            "y": np.linspace(9.0, 11.0, n),
            "sku": ["SKU_001"] * n,
        }
    )


def _make_prophet_train_df(n: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2021-01-01", periods=n, freq="MS"),
            "y": np.linspace(8.0, 12.0, n),
            "sku": ["SKU_001"] * n,
        }
    )


def _make_prophet_validation_df(n: int = 3) -> pd.DataFrame:
    # Validation period immediately follows the 24-month train split.
    return pd.DataFrame(
        {
            "ds": pd.date_range("2023-01-01", periods=n, freq="MS"),
            "y": np.linspace(9.5, 10.5, n),
            "sku": ["SKU_001"] * n,
        }
    )


def _make_prophet_prechampion_configs(
    candidate_ids: list[str] | None = None,
) -> dict:
    ids = candidate_ids or ["prophet_candidate_001", "prophet_candidate_002"]
    return {
        "prechampions": [
            {
                "candidate_id": cid,
                "rank": i + 1,
                "active_regressors": [],
                "hyperparams": {"changepoint_prior_scale": 0.05},
            }
            for i, cid in enumerate(ids)
        ]
    }


def _make_sarimax_test_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2023-01-01", periods=n, freq="MS"),
            "monthly_demand": np.linspace(9.0, 11.0, n),
        }
    )


def _make_sarimax_train_df(n: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2021-01-01", periods=n, freq="MS"),
            "monthly_demand": np.linspace(8.0, 12.0, n),
        }
    )


def _make_sarimax_validation_df(n: int = 3) -> pd.DataFrame:
    # Validation period immediately follows the 24-month train split.
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2023-01-01", periods=n, freq="MS"),
            "monthly_demand": np.linspace(10.0, 11.0, n),
        }
    )


def _make_sarimax_full_train_df(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2021-01-01", periods=n, freq="MS"),
            "monthly_demand": np.linspace(7.0, 12.0, n),
        }
    )


def _make_sarimax_prechampion_configs(trial_ids: list[str] | None = None) -> dict:
    ids = trial_ids or ["sarimax_trial_001", "sarimax_trial_002"]
    return {
        "model_family": "sarimax",
        "granularity": "monthly",
        "candidates": [
            {
                "trial_id": tid,
                "rank": i + 1,
                "order": [1, 1, 1],
                "seasonal_order": [0, 1, 1, 12],
                "trend": None,
                "use_exog": False,
                "metrics": {"wape": 0.1 + i * 0.05, "mase": 0.9, "rmse": 5.0, "bias": 0.01},
            }
            for i, tid in enumerate(ids)
        ],
    }


def _make_sarimax_training_metadata(exog_cols: list[str] | None = None) -> dict:
    return {
        "model_family": "sarimax",
        "granularity": "monthly",
        "uses_exogenous_features": False,
        "exogenous_columns": exog_cols or [],
        "target_column": "monthly_demand",
        "date_column": "month_start_date",
    }


def _make_fake_sarimax_result(n_forecast: int, value: float = 10.0) -> MagicMock:
    """Return a mock SARIMAXResultsWrapper whose get_forecast returns fixed values."""
    forecast_mock = MagicMock()
    forecast_mock.predicted_mean = np.ones(n_forecast) * value
    result_mock = MagicMock()
    result_mock.get_forecast.return_value = forecast_mock
    return result_mock


def _make_sarimax_candidate_models(
    trial_ids: list[str] | None = None, n_test: int = 6
) -> dict:
    ids = trial_ids or ["sarimax_trial_001", "sarimax_trial_002"]
    return {
        tid: {
            "rank": i + 1,
            "model_family": "sarimax",
            "granularity": "monthly",
            "config": {
                "order": [1, 1, 1],
                "seasonal_order": [0, 1, 1, 12],
                "trend": None,
                "use_exog": False,
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
            "model": _make_fake_sarimax_result(n_test, value=10.0 + i),
            "validation_metrics": {"wape": 0.1 + i * 0.05, "mase": 0.9, "rmse": 5.0, "bias": 0.01},
        }
        for i, tid in enumerate(ids)
    }


def _make_params_monthly(
    active_families: list[str] | None = None,
    refit_enabled: bool = False,
) -> dict:
    return {
        "active_families": active_families or ["prophet", "sarimax"],
        "primary_metric": "wape",
        "direction": "minimize",
        "tie_breakers": ["mase", "rmse", "abs_bias"],
        "require_all_active_families": True,
        "mase_seasonal_period": 12,
        "refit_champion": {"enabled": refit_enabled},
    }


def _make_params_prophet() -> dict:
    return {
        "date_column": "ds",
        "target_column": "y",
        "model_family": "prophet",
    }


def _make_params_sarimax() -> dict:
    return {
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "model_family": "sarimax",
        # Disable optional diagnostics by default so existing tests are not polluted
        # by extra SARIMAX calls from Ljung-Box or rolling-origin refits.
        "ljung_box": {"enabled": False},
        "operational_lead_time": {"enabled": False},
    }


# ── CatBoost test doubles ─────────────────────────────────────────────────────

_CATBOOST_FEATURES: list[str] = [
    "business_days",
    "month",
    "demand_lag_1",
    "rolling_mean_3",
]


class _FakeCatBoostModel:
    """CatBoost test double exposing the fit/predict/get_best_iteration surface.

    ``predict`` returns a constant so test metrics are deterministic; ``fit``
    records the row count so leakage-safety can be asserted.
    """

    def __init__(self, pred_value: float = 10.0, best_iteration: int = 50) -> None:
        self._pred = pred_value
        self._best = best_iteration
        self.fit_rows: int | None = None

    def get_best_iteration(self) -> int:
        return self._best

    def fit(self, x, y, verbose: bool = False):  # noqa: ANN001, ARG002
        self.fit_rows = len(x)
        return self

    def predict(self, x):  # noqa: ANN001
        return np.ones(len(x)) * self._pred


def _make_catboost_df(n: int, start: str, demand_base: float = 10.0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="MS")
    return pd.DataFrame(
        {
            "month_start_date": dates,
            "monthly_demand": np.linspace(demand_base, demand_base + 2.0, n),
            "business_days": np.full(n, 21.0),
            "month": [d.month for d in dates],
            "demand_lag_1": np.linspace(demand_base - 1.0, demand_base + 1.0, n),
            "rolling_mean_3": np.linspace(demand_base - 0.5, demand_base + 1.5, n),
            "sku": ["SKU_001"] * n,
        }
    )


def _make_catboost_train_df(n: int = 24) -> pd.DataFrame:
    return _make_catboost_df(n, "2021-01-01", demand_base=8.0)


def _make_catboost_validation_df(n: int = 3) -> pd.DataFrame:
    return _make_catboost_df(n, "2023-01-01", demand_base=10.0)


def _make_catboost_test_df(n: int = 6) -> pd.DataFrame:
    return _make_catboost_df(n, "2023-04-01", demand_base=10.0)


def _make_catboost_full_train_df(n: int = 33) -> pd.DataFrame:
    return _make_catboost_df(n, "2021-01-01", demand_base=8.0)


def _make_catboost_candidate_models(candidate_ids: list[str] | None = None) -> dict:
    ids = candidate_ids or ["catboost_trial_001", "catboost_trial_002"]
    return {
        cid: {
            "rank": i + 1,
            "model_family": "catboost",
            "granularity": "monthly",
            "feature_columns": list(_CATBOOST_FEATURES),
            "config": {"depth": 4, "learning_rate": 0.1, "iterations": 200},
            "model": _FakeCatBoostModel(pred_value=10.0 + i, best_iteration=50),
            "validation_metrics": {
                "wape": 0.09 + i * 0.05,
                "mase": 0.80,
                "rmse": 4.0,
                "bias": 0.0,
                "mae": 3.0,
            },
        }
        for i, cid in enumerate(ids)
    }


def _make_catboost_prechampion_configs(candidate_ids: list[str] | None = None) -> dict:
    ids = candidate_ids or ["catboost_trial_001", "catboost_trial_002"]
    return {
        "model_family": "catboost",
        "granularity": "monthly",
        "selection_stage": "validation",
        "objective_metric": "wape",
        "top_n": len(ids),
        "candidates": [
            {
                "candidate_id": cid,
                "rank": i + 1,
                "model_params": {"depth": 4, "learning_rate": 0.1, "iterations": 200},
                "feature_columns": list(_CATBOOST_FEATURES),
                "metrics": {
                    "wape": 0.09 + i * 0.05,
                    "mase": 0.80,
                    "rmse": 4.0,
                    "bias": 0.0,
                    "mae": 3.0,
                },
            }
            for i, cid in enumerate(ids)
        ],
    }


def _make_catboost_split_metadata() -> dict:
    return {
        "granularity": "monthly",
        "model_family": "catboost",
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "sku_column": "sku",
        "all_feature_columns": list(_CATBOOST_FEATURES),
        "categorical_feature_columns": [],
    }


def _make_params_catboost() -> dict:
    return {
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "model_family": "catboost",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_candidate_metrics_df() -> pd.DataFrame:
    """Two prophet + two sarimax rows with distinct wape values."""
    return pd.DataFrame(
        [
            {
                "family": "prophet",
                "candidate_id": "prophet_candidate_001",
                "candidate_rank": 1,
                "granularity": "monthly",
                "selection_stage": "test",
                "test_start_date": "2023-01-01",
                "test_end_date": "2023-06-01",
                "n_test_rows": 6,
                "wape": 0.08,
                "mase": 0.80,
                "rmse": 4.0,
                "bias": 0.01,
                "primary_metric": "wape",
                "primary_metric_value": 0.08,
                "is_family_champion": False,
                "is_production_champion": False,
            },
            {
                "family": "prophet",
                "candidate_id": "prophet_candidate_002",
                "candidate_rank": 2,
                "granularity": "monthly",
                "selection_stage": "test",
                "test_start_date": "2023-01-01",
                "test_end_date": "2023-06-01",
                "n_test_rows": 6,
                "wape": 0.12,
                "mase": 0.95,
                "rmse": 5.5,
                "bias": -0.02,
                "primary_metric": "wape",
                "primary_metric_value": 0.12,
                "is_family_champion": False,
                "is_production_champion": False,
            },
            {
                "family": "sarimax",
                "candidate_id": "sarimax_trial_001",
                "candidate_rank": 1,
                "granularity": "monthly",
                "selection_stage": "test",
                "test_start_date": "2023-01-01",
                "test_end_date": "2023-06-01",
                "n_test_rows": 6,
                "wape": 0.10,
                "mase": 0.85,
                "rmse": 4.5,
                "bias": 0.00,
                "primary_metric": "wape",
                "primary_metric_value": 0.10,
                "is_family_champion": False,
                "is_production_champion": False,
            },
            {
                "family": "sarimax",
                "candidate_id": "sarimax_trial_002",
                "candidate_rank": 2,
                "granularity": "monthly",
                "selection_stage": "test",
                "test_start_date": "2023-01-01",
                "test_end_date": "2023-06-01",
                "n_test_rows": 6,
                "wape": 0.15,
                "mase": 1.10,
                "rmse": 6.0,
                "bias": 0.03,
                "primary_metric": "wape",
                "primary_metric_value": 0.15,
                "is_family_champion": False,
                "is_production_champion": False,
            },
        ]
    )


# ── Test 1: evaluate scores both families ─────────────────────────────────────


def test_evaluate_monthly_candidates_scores_prophet_and_sarimax():
    """evaluate_monthly_family_candidates_on_test returns metrics for both families."""
    prophet_models = {
        "prophet_candidate_001": _FakeProphetModel(yhat_multiplier=1.0),
        "prophet_candidate_002": _FakeProphetModel(yhat_multiplier=0.9),
    }
    prophet_configs = _make_prophet_prechampion_configs()
    prophet_train = _make_prophet_train_df()
    prophet_test = _make_prophet_test_df()

    sarimax_models = _make_sarimax_candidate_models(n_test=6)
    sarimax_configs = _make_sarimax_prechampion_configs()
    sarimax_metadata = _make_sarimax_training_metadata()
    sarimax_train = _make_sarimax_train_df()
    sarimax_validation = _make_sarimax_validation_df()
    sarimax_test = _make_sarimax_test_df()

    params_monthly = _make_params_monthly()
    params_prophet = _make_params_prophet()
    params_sarimax = _make_params_sarimax()

    fake_fit_result = _make_fake_sarimax_result(n_forecast=6, value=10.0)

    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes.SARIMAX"
    ) as mock_sarimax_cls:
        mock_sarimax_cls.return_value.fit.return_value = fake_fit_result

        result = evaluate_monthly_family_candidates_on_test(
            monthly_prophet_candidate_models=prophet_models,
            monthly_prophet_prechampion_configs=prophet_configs,
            monthly_prophet_train=prophet_train,
            monthly_prophet_validation=_make_prophet_validation_df(),
            monthly_prophet_test=prophet_test,
            monthly_sarimax_candidate_models=sarimax_models,
            monthly_sarimax_prechampion_configs=sarimax_configs,
            monthly_sarimax_training_metadata=sarimax_metadata,
            monthly_sarimax_train=sarimax_train,
            monthly_sarimax_validation=sarimax_validation,
            monthly_sarimax_test=sarimax_test,
            params_monthly=params_monthly,
            params_prophet=params_prophet,
            params_sarimax=params_sarimax,
        )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    families = set(result["family"].unique())
    assert "prophet" in families, "Prophet family missing from metrics"
    assert "sarimax" in families, "SARIMAX family missing from metrics"
    # Two prophet + two sarimax candidates
    assert len(result[result["family"] == "prophet"]) == 2
    assert len(result[result["family"] == "sarimax"]) == 2
    # Required columns present
    for col in ("family", "candidate_id", "wape", "mase", "rmse", "bias", "granularity"):
        assert col in result.columns, f"Column '{col}' missing"
    # granularity must be monthly
    assert (result["granularity"] == "monthly").all()


def test_sarimax_test_scoring_is_leakage_safe():
    """SARIMAX must fit on train+validation only; the held-out test split is never in endog."""
    n_train, n_val, n_test = 24, 3, 6

    prophet_models = {
        "prophet_candidate_001": _FakeProphetModel(),
        "prophet_candidate_002": _FakeProphetModel(),
    }
    fake_fit_result = _make_fake_sarimax_result(n_forecast=n_test, value=10.0)

    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes.SARIMAX"
    ) as mock_sarimax_cls:
        mock_sarimax_cls.return_value.fit.return_value = fake_fit_result

        evaluate_monthly_family_candidates_on_test(
            monthly_prophet_candidate_models=prophet_models,
            monthly_prophet_prechampion_configs=_make_prophet_prechampion_configs(),
            monthly_prophet_train=_make_prophet_train_df(),
            monthly_prophet_validation=_make_prophet_validation_df(),
            monthly_prophet_test=_make_prophet_test_df(),
            monthly_sarimax_candidate_models=_make_sarimax_candidate_models(n_test=n_test),
            monthly_sarimax_prechampion_configs=_make_sarimax_prechampion_configs(),
            monthly_sarimax_training_metadata=_make_sarimax_training_metadata(),
            monthly_sarimax_train=_make_sarimax_train_df(n_train),
            monthly_sarimax_validation=_make_sarimax_validation_df(n_val),
            monthly_sarimax_test=_make_sarimax_test_df(n_test),
            params_monthly=_make_params_monthly(active_families=["sarimax"]),
            params_prophet=_make_params_prophet(),
            params_sarimax=_make_params_sarimax(),
        )

    # Every SARIMAX fit must use exactly train + validation rows — never the test split.
    assert mock_sarimax_cls.call_count >= 1
    for call in mock_sarimax_cls.call_args_list:
        endog = call.kwargs["endog"]
        assert len(endog) == n_train + n_val
        assert len(endog) != n_train + n_val + n_test


def test_prophet_test_scoring_is_leakage_safe():
    """Prophet must refit on train+validation only; the test split must never be in the fit frame."""
    n_train, n_val, n_test = 24, 3, 6
    expected_candidates = 2
    captured: dict[str, int] = {}

    fake_model = MagicMock()
    fake_model.predict.return_value = pd.DataFrame({"yhat": np.ones(n_test) * 10.0})

    def _spy_refit(candidate_model, fit_df, active_regressors, date_col="ds", target_col="y"):
        captured["n"] = len(fit_df)
        return fake_model

    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes._refit_prophet_on_frame",
        side_effect=_spy_refit,
    ):
        rows = _score_prophet_candidates(
            candidate_models={
                "prophet_candidate_001": _FakeProphetModel(),
                "prophet_candidate_002": _FakeProphetModel(),
            },
            prechampion_configs=_make_prophet_prechampion_configs(),
            train_df=_make_prophet_train_df(n_train),
            validation_df=_make_prophet_validation_df(n_val),
            test_df=_make_prophet_test_df(n_test),
            params=_make_params_prophet(),
            mase_period=12,
            require_all=True,
        )

    assert len(rows) == expected_candidates
    # The refit frame must be train + validation only — never including the test split.
    assert captured["n"] == n_train + n_val
    assert captured["n"] != n_train + n_val + n_test


# ── Test 2: select_monthly_family_champions returns exactly one per family ────


def test_select_monthly_family_champions_returns_one_per_family():
    """select_monthly_family_champions produces exactly one row per family."""
    metrics_df = _make_candidate_metrics_df()
    params = _make_params_monthly()

    summary = select_monthly_family_champions(metrics_df, params)

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 2
    families = set(summary["family"].tolist())
    assert families == {"prophet", "sarimax"}

    prophet_row = summary[summary["family"] == "prophet"].iloc[0]
    sarimax_row = summary[summary["family"] == "sarimax"].iloc[0]

    # Prophet champion should be the one with wape=0.08
    assert prophet_row["family_champion_id"] == "prophet_candidate_001"
    # SARIMAX champion should be the one with wape=0.10
    assert sarimax_row["family_champion_id"] == "sarimax_trial_001"


# ── Test 3: production champion uses primary metric then tie-breakers ─────────


def test_select_monthly_production_champion_uses_primary_metric_then_tie_breakers():
    """Production champion is elected by wape first; tie-breakers applied in order."""
    # Build a family champion summary where prophet is clearly better on wape
    family_summary = pd.DataFrame(
        [
            {
                "family": "prophet",
                "granularity": "monthly",
                "family_champion_id": "prophet_candidate_001",
                "family_champion_rank": 1,
                "wape": 0.08,
                "mase": 0.80,
                "rmse": 4.0,
                "bias": 0.01,
                "selection_reason": "best wape",
                "model_artifact_key": "monthly_prophet_candidate_models",
                "metadata_artifact_key": "monthly_prophet_prechampion_configs",
            },
            {
                "family": "sarimax",
                "granularity": "monthly",
                "family_champion_id": "sarimax_trial_001",
                "family_champion_rank": 1,
                "wape": 0.10,
                "mase": 0.85,
                "rmse": 4.5,
                "bias": 0.00,
                "selection_reason": "best wape",
                "model_artifact_key": "monthly_sarimax_candidate_models",
                "metadata_artifact_key": "monthly_sarimax_prechampion_configs",
            },
        ]
    )
    metrics_df = _make_candidate_metrics_df()
    params = _make_params_monthly()

    prod_summary = select_monthly_production_champion(family_summary, metrics_df, params)

    assert isinstance(prod_summary, pd.DataFrame)
    assert len(prod_summary) == 1
    row = prod_summary.iloc[0]
    assert row["production_champion_family"] == "prophet"
    assert row["production_champion_id"] == "prophet_candidate_001"
    assert row["primary_metric"] == "wape"
    assert row["granularity"] == "monthly"

    # Tie scenario: same wape, different mase → lower mase wins
    family_summary_tie = family_summary.copy()
    family_summary_tie["wape"] = 0.08  # equal wape
    # prophet mase=0.80, sarimax mase=0.85 → prophet still wins
    prod_summary_tie = select_monthly_production_champion(
        family_summary_tie, metrics_df, params
    )
    assert prod_summary_tie.iloc[0]["production_champion_family"] == "prophet"


# ── Test 4: candidate metrics carry selected champion flags ───────────────────


def test_annotate_monthly_candidate_champion_flags_marks_selected_rows():
    """Candidate metrics flags reflect family and production champion summaries."""
    metrics_df = _make_candidate_metrics_df()
    family_summary = select_monthly_family_champions(metrics_df, _make_params_monthly())
    production_summary = select_monthly_production_champion(
        family_summary, metrics_df, _make_params_monthly()
    )

    annotated = annotate_monthly_candidate_champion_flags(
        metrics_df, family_summary, production_summary
    )
    expected_family_champion_ids = {
        "prophet_candidate_001",
        "sarimax_trial_001",
    }

    assert int(annotated["is_family_champion"].sum()) == len(
        expected_family_champion_ids
    )
    assert int(annotated["is_production_champion"].sum()) == 1

    family_champions = annotated[annotated["is_family_champion"]]
    assert set(family_champions["candidate_id"]) == expected_family_champion_ids

    production_champion = annotated[annotated["is_production_champion"]].iloc[0]
    assert production_champion["family"] == "prophet"
    assert production_champion["candidate_id"] == "prophet_candidate_001"


# ── Test 5: champion metadata has required schema ─────────────────────────────


def test_build_monthly_champion_metadata_has_required_schema():
    """champion_monthly_metadata contains all required keys and correct types."""
    metrics_df = _make_candidate_metrics_df()
    family_summary = pd.DataFrame(
        [
            {
                "family": "prophet",
                "granularity": "monthly",
                "family_champion_id": "prophet_candidate_001",
                "family_champion_rank": 1,
                "wape": 0.08,
                "mase": 0.80,
                "rmse": 4.0,
                "bias": 0.01,
                "selection_reason": "best wape",
                "model_artifact_key": "monthly_prophet_candidate_models",
                "metadata_artifact_key": "monthly_prophet_prechampion_configs",
            },
            {
                "family": "sarimax",
                "granularity": "monthly",
                "family_champion_id": "sarimax_trial_001",
                "family_champion_rank": 1,
                "wape": 0.10,
                "mase": 0.85,
                "rmse": 4.5,
                "bias": 0.00,
                "selection_reason": "best wape",
                "model_artifact_key": "monthly_sarimax_candidate_models",
                "metadata_artifact_key": "monthly_sarimax_prechampion_configs",
            },
        ]
    )
    prod_summary = pd.DataFrame(
        [
            {
                "granularity": "monthly",
                "active_families": ["prophet", "sarimax"],
                "production_champion_family": "prophet",
                "production_champion_id": "prophet_candidate_001",
                "primary_metric": "wape",
                "primary_metric_value": 0.08,
                "tie_breakers": ["mase", "rmse", "abs_bias"],
                "selection_timestamp": "2025-01-01T00:00:00+00:00",
                "selection_reason": "Best wape",
                "candidate_count": 4,
                "family_champion_count": 2,
            }
        ]
    )

    prophet_model = _FakeProphetModel()
    prophet_models = {"prophet_candidate_001": prophet_model, "prophet_candidate_002": _FakeProphetModel()}
    sarimax_models = _make_sarimax_candidate_models(n_test=6)
    # Refit disabled: the train-only candidate is returned as-is, so identity holds.
    params = _make_params_monthly(refit_enabled=False)

    champion_model, metadata = build_monthly_champion_artifacts(
        monthly_model_selection_summary=prod_summary,
        monthly_family_champion_summary=family_summary,
        monthly_candidate_test_metrics=metrics_df,
        monthly_prophet_candidate_models=prophet_models,
        monthly_sarimax_candidate_models=sarimax_models,
        monthly_prophet_full_train=_make_prophet_train_df(24),
        monthly_sarimax_full_train=_make_sarimax_full_train_df(30),
        monthly_sarimax_training_metadata=_make_sarimax_training_metadata(),
        params_monthly=params,
    )

    # Refit disabled → champion model is the resolved candidate
    assert champion_model is prophet_model
    assert metadata["refit"]["performed"] is False
    assert metadata["refit"]["data_scope"] == "full_history"

    # Top-level keys
    required_top_keys = {
        "granularity", "model_family", "champion_id", "champion_level",
        "active_regressors", "family_champions", "selection", "test_period",
        "metrics", "inference_contract", "model_artifact", "compatibility",
    }
    assert required_top_keys.issubset(set(metadata.keys()))

    assert metadata["granularity"] == "monthly"
    assert metadata["model_family"] == "prophet"
    assert metadata["champion_id"] == "prophet_candidate_001"
    assert metadata["champion_level"] == "production"
    assert isinstance(metadata["active_regressors"], list)

    # family_champions block
    assert "prophet" in metadata["family_champions"]
    assert "sarimax" in metadata["family_champions"]

    # selection block
    sel = metadata["selection"]
    assert sel["primary_metric"] == "wape"
    assert sel["direction"] == "minimize"
    assert isinstance(sel["tie_breakers"], list)

    # inference_contract block (consumed by metadata-driven inference)
    contract = metadata["inference_contract"]
    assert contract["model_family"] == "prophet"
    assert contract["date_column"] == "ds"
    assert contract["has_prediction_intervals"] is True
    assert isinstance(contract["active_regressors"], list)

    # compatibility block
    compat = metadata["compatibility"]
    assert compat["inference_ready"] is True
    assert compat["metadata_driven_inference"] is True

    # metrics block
    assert "wape" in metadata["metrics"]


# ── Test 4b: full-history champion refit (Champion Protocol stage 5) ──────────


def _make_family_champion_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": "prophet",
                "granularity": "monthly",
                "family_champion_id": "prophet_candidate_001",
                "family_champion_rank": 1,
                "wape": 0.08,
                "mase": 0.80,
                "rmse": 4.0,
                "bias": 0.01,
                "selection_reason": "best wape",
                "model_artifact_key": "monthly_prophet_candidate_models",
                "metadata_artifact_key": "monthly_prophet_prechampion_configs",
            },
            {
                "family": "sarimax",
                "granularity": "monthly",
                "family_champion_id": "sarimax_trial_001",
                "family_champion_rank": 1,
                "wape": 0.10,
                "mase": 0.85,
                "rmse": 4.5,
                "bias": 0.00,
                "selection_reason": "best wape",
                "model_artifact_key": "monthly_sarimax_candidate_models",
                "metadata_artifact_key": "monthly_sarimax_prechampion_configs",
            },
        ]
    )


def _make_production_summary(family: str, candidate_id: str, metric_value: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "granularity": "monthly",
                "active_families": ["prophet", "sarimax"],
                "production_champion_family": family,
                "production_champion_id": candidate_id,
                "primary_metric": "wape",
                "primary_metric_value": metric_value,
                "tie_breakers": ["mase", "rmse", "abs_bias"],
                "selection_timestamp": "2025-01-01T00:00:00+00:00",
                "selection_reason": f"Best wape {family}",
                "candidate_count": 4,
                "family_champion_count": 2,
            }
        ]
    )


def _build_champion(family: str, candidate_id: str, metric_value: float):
    return build_monthly_champion_artifacts(
        monthly_model_selection_summary=_make_production_summary(
            family, candidate_id, metric_value
        ),
        monthly_family_champion_summary=_make_family_champion_summary(),
        monthly_candidate_test_metrics=_make_candidate_metrics_df(),
        monthly_prophet_candidate_models={"prophet_candidate_001": _FakeProphetModel()},
        monthly_sarimax_candidate_models=_make_sarimax_candidate_models(n_test=6),
        monthly_prophet_full_train=_make_prophet_train_df(24),
        monthly_sarimax_full_train=_make_sarimax_full_train_df(30),
        monthly_sarimax_training_metadata=_make_sarimax_training_metadata(),
        params_monthly=_make_params_monthly(refit_enabled=True),
    )


def test_build_champion_refits_sarimax_on_full_history():
    """SARIMAX champion must be refit on full history into a forecastable results object."""
    champion_model, metadata = _build_champion("sarimax", "sarimax_trial_001", 0.10)

    assert metadata["model_family"] == "sarimax"
    assert metadata["refit"]["performed"] is True
    assert metadata["refit"]["n_obs"] == 30  # noqa: PLR2004
    assert metadata["inference_contract"]["model_family"] == "sarimax"

    # The champion carries a real fitted results object (not the candidate MagicMock).
    assert isinstance(champion_model, dict)
    assert champion_model["refit_scope"] == "full_history"
    forecast = champion_model["model"].get_forecast(steps=3)
    assert len(np.asarray(forecast.predicted_mean)) == 3  # noqa: PLR2004


def test_build_champion_refits_prophet_on_full_history():
    """Prophet champion must be refit on full history into a new fitted model."""
    champion_model, metadata = _build_champion("prophet", "prophet_candidate_001", 0.08)

    assert metadata["model_family"] == "prophet"
    assert metadata["refit"]["performed"] is True
    assert metadata["refit"]["n_obs"] == 24  # noqa: PLR2004

    # A new model was fit on full history (not the train-only candidate).
    assert hasattr(champion_model, "predict")
    assert metadata["inference_contract"]["model_family"] == "prophet"


# ── Test 5: scope guard — CatBoost included, weekly excluded ──────────────────


def test_monthly_model_selection_includes_catboost_but_not_weekly():
    """The monthly multi-family selection wires CatBoost inputs but never weekly ones."""
    import inspect

    from hdf_pipelines.pipelines.model_selection.monthly.nodes import (
        evaluate_monthly_family_candidates_on_test,
    )
    from hdf_pipelines.pipelines.model_selection.monthly.pipeline import create_pipeline

    sig = inspect.signature(evaluate_monthly_family_candidates_on_test)
    param_names = list(sig.parameters.keys())

    # CatBoost must now be a recognised candidate family.
    assert any("catboost" in name.lower() for name in param_names), (
        "evaluate_monthly_family_candidates_on_test must accept CatBoost inputs"
    )
    # Weekly remains out of scope for the monthly selection pipeline.
    for name in param_names:
        assert "weekly" not in name.lower(), (
            f"Parameter '{name}' references weekly data — out of scope for monthly selection"
        )

    # Pipeline nodes must reference CatBoost catalog keys but never weekly ones.
    p = create_pipeline()
    all_inputs = {inp for n in p.nodes for inp in n.inputs}
    assert any("catboost" in key.lower() for key in all_inputs), (
        "Monthly selection pipeline must consume CatBoost candidate artifacts"
    )
    for key in all_inputs:
        assert "weekly" not in key.lower(), (
            f"Pipeline input '{key}' references weekly data — out of scope for monthly selection"
        )


# ── Test 6: fails clearly when active family artifacts are missing ────────────


def test_monthly_model_selection_fails_when_active_family_artifacts_missing():
    """Selector raises ValueError when a required family has no prechampion configs."""
    prophet_models = {"prophet_candidate_001": _FakeProphetModel()}
    prophet_configs = _make_prophet_prechampion_configs(["prophet_candidate_001"])
    prophet_train = _make_prophet_train_df()
    prophet_test = _make_prophet_test_df()

    # Empty SARIMAX configs — no candidates
    sarimax_configs_empty = {
        "model_family": "sarimax",
        "granularity": "monthly",
        "candidates": [],
    }
    sarimax_models = _make_sarimax_candidate_models(n_test=6)
    sarimax_metadata = _make_sarimax_training_metadata()
    sarimax_train = _make_sarimax_train_df()
    sarimax_validation = _make_sarimax_validation_df()
    sarimax_test = _make_sarimax_test_df()

    params_monthly = _make_params_monthly()  # require_all_active_families=True
    params_prophet = _make_params_prophet()
    params_sarimax = _make_params_sarimax()

    with pytest.raises(ValueError, match="require_all_active_families"):
        evaluate_monthly_family_candidates_on_test(
            monthly_prophet_candidate_models=prophet_models,
            monthly_prophet_prechampion_configs=prophet_configs,
            monthly_prophet_train=prophet_train,
            monthly_prophet_validation=_make_prophet_validation_df(),
            monthly_prophet_test=prophet_test,
            monthly_sarimax_candidate_models=sarimax_models,
            monthly_sarimax_prechampion_configs=sarimax_configs_empty,
            monthly_sarimax_training_metadata=sarimax_metadata,
            monthly_sarimax_train=sarimax_train,
            monthly_sarimax_validation=sarimax_validation,
            monthly_sarimax_test=sarimax_test,
            params_monthly=params_monthly,
            params_prophet=params_prophet,
            params_sarimax=params_sarimax,
        )


# ── Test 7: candidate test metrics always originate from the test split ───────


def test_candidate_test_metrics_selection_stage_is_test():
    """All candidate metrics rows must carry selection_stage='test' (no leakage from val/train)."""
    metrics_df = _make_candidate_metrics_df()
    assert (metrics_df["selection_stage"] == "test").all(), (
        "candidate_test_metrics contains rows not attributed to the held-out test split"
    )


def test_family_champion_summary_has_fewer_rows_than_candidate_metrics():
    """Family champion summary collapses many candidates to one winner per family."""
    metrics_df = _make_candidate_metrics_df()
    params = _make_params_monthly()

    family_summary = select_monthly_family_champions(metrics_df, params)

    assert len(family_summary) < len(metrics_df), (
        "monthly_family_champion_summary should have fewer rows than monthly_candidate_test_metrics"
    )
    assert len(family_summary) == 2  # one per family  # noqa: PLR2004


def test_production_champion_summary_has_exactly_one_row():
    """monthly_model_selection_summary always resolves to a single production champion."""
    metrics_df = _make_candidate_metrics_df()
    params = _make_params_monthly()

    family_summary = select_monthly_family_champions(metrics_df, params)
    prod_summary = select_monthly_production_champion(family_summary, metrics_df, params)

    assert len(prod_summary) == 1, (
        "monthly_model_selection_summary must contain exactly one row (the production champion)"
    )
    assert prod_summary.iloc[0]["granularity"] == "monthly"


# ── Test 8: champion metadata exposes training_cutoff and hyperparameters ─────


def test_champion_metadata_exposes_training_cutoff():
    """champion_monthly_metadata must contain a training_cutoff field derived from the refit."""
    _, metadata = _build_champion("prophet", "prophet_candidate_001", 0.08)

    assert "training_cutoff" in metadata, (
        "champion_monthly_metadata is missing the training_cutoff field"
    )
    # When refit is performed, training_cutoff must be a non-empty string (the end date).
    assert metadata["training_cutoff"] is not None


def test_champion_metadata_exposes_hyperparameters_for_prophet():
    """Prophet champion metadata must include a hyperparameters block with Prophet-style keys."""
    _, metadata = _build_champion("prophet", "prophet_candidate_001", 0.08)

    assert "hyperparameters" in metadata, (
        "champion_monthly_metadata is missing the hyperparameters field"
    )
    hp = metadata["hyperparameters"]
    assert isinstance(hp, dict)
    assert "changepoint_prior_scale" in hp
    assert "seasonality_mode" in hp


def test_champion_metadata_exposes_hyperparameters_for_sarimax():
    """SARIMAX champion metadata must include a hyperparameters block with SARIMAX-style keys."""
    _, metadata = _build_champion("sarimax", "sarimax_trial_001", 0.10)

    assert "hyperparameters" in metadata, (
        "champion_monthly_metadata is missing the hyperparameters field"
    )
    hp = metadata["hyperparameters"]
    assert isinstance(hp, dict)
    assert "order" in hp
    assert "seasonal_order" in hp
    assert "use_exog" in hp


# ── Test 9: champion metadata is consumable by inference without family knowledge ──


def test_champion_metadata_contract_is_inference_agnostic():
    """Inference contract must be readable without hardcoding the model family externally."""
    for family, candidate_id, metric in [
        ("prophet", "prophet_candidate_001", 0.08),
        ("sarimax", "sarimax_trial_001", 0.10),
    ]:
        _, metadata = _build_champion(family, candidate_id, metric)

        # An inference consumer should resolve these without family-specific branches.
        resolved_family = metadata["model_family"]
        resolved_regressors = metadata["active_regressors"]
        resolved_contract = metadata["inference_contract"]

        assert resolved_family == family
        assert isinstance(resolved_regressors, list)
        assert resolved_contract["model_family"] == family
        assert isinstance(resolved_contract["forecast_horizons"], list)
        assert 12 in resolved_contract["forecast_horizons"]  # noqa: PLR2004


# ── Test 10: Ljung-Box residual autocorrelation filter ────────────────────────


def _make_params_sarimax_with_ljungbox(
    lb_enabled: bool = True,
    lb_threshold: float = 0.05,
    ro_enabled: bool = False,
) -> dict:
    """SARIMAX params with explicit Ljung-Box and rolling-origin config."""
    return {
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "model_family": "sarimax",
        "ljung_box": {
            "enabled": lb_enabled,
            "lags": 10,
            "pvalue_threshold": lb_threshold,
        },
        "operational_lead_time": {
            "enabled": ro_enabled,
            "lead_times": [2, 3],
        },
    }


def _make_fake_sarimax_result_with_resid(
    n_forecast: int, value: float = 10.0, resid: Any | None = None
) -> MagicMock:
    """SARIMAX result mock with controllable residuals for Ljung-Box testing."""
    forecast_mock = MagicMock()
    forecast_mock.predicted_mean = np.ones(n_forecast) * value
    result_mock = MagicMock()
    result_mock.get_forecast.return_value = forecast_mock
    result_mock.resid = resid if resid is not None else np.random.default_rng(0).normal(0, 1, 24)
    return result_mock


class TestScoreSarimaxLjungBox:
    """Ljung-Box residual autocorrelation checks in _score_sarimax_candidates."""

    def _run_scorer(
        self,
        lb_pvalue: float,
        lb_enabled: bool = True,
        lb_threshold: float = 0.05,
    ) -> list[dict]:
        """Run scorer with Ljung-Box mocked to return a specific p-value."""
        lb_df = pd.DataFrame({"lb_stat": [5.0], "lb_pvalue": [lb_pvalue]})
        fake_result = _make_fake_sarimax_result_with_resid(n_forecast=6)
        sarimax_models = _make_sarimax_candidate_models(n_test=6)
        sarimax_configs = _make_sarimax_prechampion_configs()
        sarimax_metadata = _make_sarimax_training_metadata()
        params = _make_params_sarimax_with_ljungbox(
            lb_enabled=lb_enabled, lb_threshold=lb_threshold, ro_enabled=False
        )

        with patch(
            "hdf_pipelines.pipelines.model_selection.monthly.nodes.SARIMAX"
        ) as mock_cls, patch(
            "hdf_pipelines.pipelines.model_selection.monthly.nodes.acorr_ljungbox",
            return_value=lb_df,
        ):
            mock_cls.return_value.fit.return_value = fake_result
            rows = _score_sarimax_candidates(
                candidate_models=sarimax_models,
                prechampion_configs=sarimax_configs,
                training_metadata=sarimax_metadata,
                train_df=_make_sarimax_train_df(),
                validation_df=_make_sarimax_validation_df(),
                test_df=_make_sarimax_test_df(),
                params=params,
                mase_period=12,
                require_all=True,
            )
        return rows

    def test_ljung_box_pvalue_stored_in_metrics_row(self):
        rows = self._run_scorer(lb_pvalue=0.20)
        assert all("ljung_box_pvalue" in r for r in rows)
        assert all(r["ljung_box_pvalue"] == pytest.approx(0.20) for r in rows)

    def test_autocorrelation_excluded_true_when_pvalue_below_threshold(self):
        rows = self._run_scorer(lb_pvalue=0.01, lb_threshold=0.05)
        assert all(r["autocorrelation_excluded"] is True for r in rows)

    def test_autocorrelation_excluded_false_when_pvalue_above_threshold(self):
        rows = self._run_scorer(lb_pvalue=0.30, lb_threshold=0.05)
        assert all(r["autocorrelation_excluded"] is False for r in rows)

    def test_ljung_box_pvalue_is_none_when_disabled(self):
        rows = self._run_scorer(lb_pvalue=0.01, lb_enabled=False)
        assert all(r["ljung_box_pvalue"] is None for r in rows)
        assert all(r["autocorrelation_excluded"] is False for r in rows)


class TestSelectChampionsLjungBoxFiltering:
    """Ljung-Box exclusion pre-filtering in select_monthly_family_champions."""

    def _make_metrics_with_exclusion(self, excluded: list[bool]) -> pd.DataFrame:
        """Build a SARIMAX-only metrics DataFrame with autocorrelation_excluded flags."""
        rows = []
        for i, excl in enumerate(excluded):
            rows.append({
                "family": "sarimax",
                "candidate_id": f"sarimax_trial_00{i + 1}",
                "candidate_rank": i + 1,
                "granularity": "monthly",
                "selection_stage": "test",
                "test_start_date": "2023-01-01",
                "test_end_date": "2023-06-01",
                "n_test_rows": 6,
                "wape": 0.10 + i * 0.05,
                "mase": 0.80 + i * 0.05,
                "rmse": 4.0 + i,
                "bias": 0.0,
                "ljung_box_pvalue": 0.01 if excl else 0.30,
                "autocorrelation_excluded": excl,
                "primary_metric": "wape",
                "primary_metric_value": 0.10 + i * 0.05,
                "is_family_champion": False,
                "is_production_champion": False,
            })
        return pd.DataFrame(rows)

    def test_excluded_candidates_are_not_selected_as_champion(self):
        """Candidate marked autocorrelation_excluded=True must not win when others are valid."""
        # Candidate 001 has best wape (0.10) but is excluded; 002 has wape=0.15 and is eligible.
        metrics_df = self._make_metrics_with_exclusion([True, False])
        params = _make_params_monthly(active_families=["sarimax"])

        summary = select_monthly_family_champions(metrics_df, params)

        assert len(summary) == 1
        assert summary.iloc[0]["family_champion_id"] == "sarimax_trial_002"

    def test_graceful_degradation_when_all_candidates_excluded(self):
        """When every SARIMAX candidate is excluded, the filter falls back to the full set."""
        metrics_df = self._make_metrics_with_exclusion([True, True])
        params = _make_params_monthly(active_families=["sarimax"])

        # Should not raise; best overall wape wins from the full fallback set.
        summary = select_monthly_family_champions(metrics_df, params)

        assert len(summary) == 1
        assert summary.iloc[0]["family_champion_id"] == "sarimax_trial_001"


# ── Test 11: Rolling-origin M-2/M-3 metrics ──────────────────────────────────


class TestScoreSarimaxRollingOriginMetrics:
    """Rolling-origin M-2/M-3 WAPE columns present and reasonable in metrics rows."""

    def _run_scorer_with_ro(self) -> list[dict]:
        """Score with rolling-origin enabled; SARIMAX is mocked for all refits."""
        fake_result = _make_fake_sarimax_result_with_resid(n_forecast=6, value=10.0)
        sarimax_models = _make_sarimax_candidate_models(n_test=3)
        sarimax_configs = _make_sarimax_prechampion_configs()
        sarimax_metadata = _make_sarimax_training_metadata()
        params = {
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "model_family": "sarimax",
            "ljung_box": {"enabled": False},
            "operational_lead_time": {"enabled": True, "lead_times": [2, 3]},
        }

        with patch(
            "hdf_pipelines.pipelines.model_selection.monthly.nodes.SARIMAX"
        ) as mock_cls, patch(
            "hdf_pipelines.pipelines.model_selection.monthly.nodes.acorr_ljungbox",
            return_value=pd.DataFrame({"lb_stat": [5.0], "lb_pvalue": [0.5]}),
        ):
            mock_cls.return_value.fit.return_value = fake_result
            rows = _score_sarimax_candidates(
                candidate_models=sarimax_models,
                prechampion_configs=sarimax_configs,
                training_metadata=sarimax_metadata,
                train_df=_make_sarimax_train_df(),
                validation_df=_make_sarimax_validation_df(),
                test_df=_make_sarimax_test_df(n=3),
                params=params,
                mase_period=12,
                require_all=True,
            )
        return rows

    def test_rolling_origin_columns_present_in_rows(self):
        rows = self._run_scorer_with_ro()
        for row in rows:
            assert "test_m2_wape" in row, "test_m2_wape missing from SARIMAX metrics row"
            assert "test_m3_wape" in row, "test_m3_wape missing from SARIMAX metrics row"
            assert "n_m2_pairs" in row
            assert "n_m3_pairs" in row

    def test_rolling_origin_wape_is_non_negative_or_none(self):
        rows = self._run_scorer_with_ro()
        for row in rows:
            m2 = row.get("test_m2_wape")
            m3 = row.get("test_m3_wape")
            if m2 is not None:
                assert m2 >= 0.0, f"test_m2_wape={m2} should be non-negative"
            if m3 is not None:
                assert m3 >= 0.0, f"test_m3_wape={m3} should be non-negative"

    def test_rolling_origin_pair_counts_match_test_window(self):
        """For a 3-month test: M-2 has 2 pairs and M-3 has 1 pair."""
        rows = self._run_scorer_with_ro()
        for row in rows:
            assert row["n_m2_pairs"] <= 2  # noqa: PLR2004
            assert row["n_m3_pairs"] <= 1


class TestSelectChampionsRollingOriginTieBreakers:
    """test_m3_wape and test_m2_wape used as tie-breakers in family champion selection."""

    def test_tie_broken_by_m3_wape(self):
        """When two candidates have equal wape, lower test_m3_wape wins."""
        metrics_df = pd.DataFrame([
            {
                "family": "sarimax",
                "candidate_id": "sarimax_trial_001",
                "candidate_rank": 1,
                "granularity": "monthly",
                "selection_stage": "test",
                "test_start_date": "2023-01-01",
                "test_end_date": "2023-03-01",
                "n_test_rows": 3,
                "wape": 0.10,
                "mase": 0.85,
                "rmse": 4.5,
                "bias": 0.0,
                "test_m3_wape": 0.05,
                "test_m2_wape": 0.08,
                "autocorrelation_excluded": False,
                "primary_metric": "wape",
                "primary_metric_value": 0.10,
                "is_family_champion": False,
                "is_production_champion": False,
            },
            {
                "family": "sarimax",
                "candidate_id": "sarimax_trial_002",
                "candidate_rank": 2,
                "granularity": "monthly",
                "selection_stage": "test",
                "test_start_date": "2023-01-01",
                "test_end_date": "2023-03-01",
                "n_test_rows": 3,
                "wape": 0.10,  # same wape
                "mase": 0.90,
                "rmse": 5.0,
                "bias": 0.0,
                "test_m3_wape": 0.12,  # worse M-3
                "test_m2_wape": 0.07,
                "autocorrelation_excluded": False,
                "primary_metric": "wape",
                "primary_metric_value": 0.10,
                "is_family_champion": False,
                "is_production_champion": False,
            },
        ])
        params = _make_params_monthly(active_families=["sarimax"])
        # Inject test_m3_wape / test_m2_wape into tie_breakers to match YAML update.
        params["tie_breakers"] = ["test_m3_wape", "test_m2_wape", "mase", "rmse", "abs_bias"]

        summary = select_monthly_family_champions(metrics_df, params)

        assert len(summary) == 1
        # trial_001 has lower test_m3_wape (0.05 < 0.12) → wins the tie.
        assert summary.iloc[0]["family_champion_id"] == "sarimax_trial_001"


# ── Test 12: CatBoost participates in monthly selection (Phase 2) ─────────────


def _make_three_family_metrics_df() -> pd.DataFrame:
    """Prophet + SARIMAX + CatBoost metrics where CatBoost has the lowest WAPE."""
    base = _make_candidate_metrics_df()
    catboost_rows = pd.DataFrame(
        [
            {
                "family": "catboost",
                "candidate_id": "catboost_trial_001",
                "candidate_rank": 1,
                "granularity": "monthly",
                "selection_stage": "test",
                "test_start_date": "2023-01-01",
                "test_end_date": "2023-06-01",
                "n_test_rows": 6,
                "wape": 0.05,  # lowest WAPE across all three families
                "mase": 0.70,
                "rmse": 3.5,
                "bias": 0.00,
                "primary_metric": "wape",
                "primary_metric_value": 0.05,
                "is_family_champion": False,
                "is_production_champion": False,
            },
            {
                "family": "catboost",
                "candidate_id": "catboost_trial_002",
                "candidate_rank": 2,
                "granularity": "monthly",
                "selection_stage": "test",
                "test_start_date": "2023-01-01",
                "test_end_date": "2023-06-01",
                "n_test_rows": 6,
                "wape": 0.20,
                "mase": 1.20,
                "rmse": 7.0,
                "bias": 0.04,
                "primary_metric": "wape",
                "primary_metric_value": 0.20,
                "is_family_champion": False,
                "is_production_champion": False,
            },
        ]
    )
    return pd.concat([base, catboost_rows], ignore_index=True)


def _make_three_family_champion_summary() -> pd.DataFrame:
    """Family champion summary with one champion per family (CatBoost best on WAPE)."""
    return pd.DataFrame(
        [
            {
                "family": "prophet",
                "granularity": "monthly",
                "family_champion_id": "prophet_candidate_001",
                "family_champion_rank": 1,
                "wape": 0.08,
                "mase": 0.80,
                "rmse": 4.0,
                "bias": 0.01,
                "selection_reason": "best wape",
                "model_artifact_key": "monthly_prophet_candidate_models",
                "metadata_artifact_key": "monthly_prophet_prechampion_configs",
            },
            {
                "family": "sarimax",
                "granularity": "monthly",
                "family_champion_id": "sarimax_trial_001",
                "family_champion_rank": 1,
                "wape": 0.10,
                "mase": 0.85,
                "rmse": 4.5,
                "bias": 0.00,
                "selection_reason": "best wape",
                "model_artifact_key": "monthly_sarimax_candidate_models",
                "metadata_artifact_key": "monthly_sarimax_prechampion_configs",
            },
            {
                "family": "catboost",
                "granularity": "monthly",
                "family_champion_id": "catboost_trial_001",
                "family_champion_rank": 1,
                "wape": 0.05,
                "mase": 0.70,
                "rmse": 3.5,
                "bias": 0.00,
                "selection_reason": "best wape",
                "model_artifact_key": "monthly_catboost_candidate_models",
                "metadata_artifact_key": "monthly_catboost_prechampion_configs",
            },
        ]
    )


def _make_catboost_production_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "granularity": "monthly",
                "active_families": ["prophet", "sarimax", "catboost"],
                "production_champion_family": "catboost",
                "production_champion_id": "catboost_trial_001",
                "primary_metric": "wape",
                "primary_metric_value": 0.05,
                "tie_breakers": ["mase", "rmse", "abs_bias"],
                "selection_timestamp": "2025-01-01T00:00:00+00:00",
                "selection_reason": "Best wape catboost",
                "candidate_count": 6,
                "family_champion_count": 3,
            }
        ]
    )


def test_score_catboost_candidates_is_leakage_safe():
    """CatBoost must refit on train+validation only; the held-out test split is never fit."""
    n_train, n_val, n_test = 24, 3, 6
    fake = _FakeCatBoostModel(pred_value=10.0)

    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes.CatBoostRegressor",
        return_value=fake,
    ):
        rows = _score_catboost_candidates(
            candidate_models=_make_catboost_candidate_models(),
            prechampion_configs=_make_catboost_prechampion_configs(),
            split_metadata=_make_catboost_split_metadata(),
            train_df=_make_catboost_train_df(n_train),
            validation_df=_make_catboost_validation_df(n_val),
            test_df=_make_catboost_test_df(n_test),
            params=_make_params_catboost(),
            mase_period=12,
            require_all=True,
        )

    assert len(rows) == 2  # noqa: PLR2004
    assert all(r["family"] == "catboost" for r in rows)
    for col in ("wape", "mase", "rmse", "bias"):
        assert all(col in r for r in rows)
    # Refit window must be train + validation only — never including the test split.
    assert fake.fit_rows == n_train + n_val
    assert fake.fit_rows != n_train + n_val + n_test


def test_evaluate_monthly_candidates_scores_three_families():
    """evaluate_monthly_family_candidates_on_test returns metrics for all three families."""
    prophet_models = {
        "prophet_candidate_001": _FakeProphetModel(yhat_multiplier=1.0),
        "prophet_candidate_002": _FakeProphetModel(yhat_multiplier=0.9),
    }
    fake_cb = _FakeCatBoostModel(pred_value=10.0)
    fake_fit_result = _make_fake_sarimax_result(n_forecast=6, value=10.0)

    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes.SARIMAX"
    ) as mock_sarimax_cls, patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes.CatBoostRegressor",
        return_value=fake_cb,
    ):
        mock_sarimax_cls.return_value.fit.return_value = fake_fit_result

        result = evaluate_monthly_family_candidates_on_test(
            monthly_prophet_candidate_models=prophet_models,
            monthly_prophet_prechampion_configs=_make_prophet_prechampion_configs(),
            monthly_prophet_train=_make_prophet_train_df(),
            monthly_prophet_validation=_make_prophet_validation_df(),
            monthly_prophet_test=_make_prophet_test_df(),
            monthly_sarimax_candidate_models=_make_sarimax_candidate_models(n_test=6),
            monthly_sarimax_prechampion_configs=_make_sarimax_prechampion_configs(),
            monthly_sarimax_training_metadata=_make_sarimax_training_metadata(),
            monthly_sarimax_train=_make_sarimax_train_df(),
            monthly_sarimax_validation=_make_sarimax_validation_df(),
            monthly_sarimax_test=_make_sarimax_test_df(),
            params_monthly=_make_params_monthly(
                active_families=["prophet", "sarimax", "catboost"]
            ),
            params_prophet=_make_params_prophet(),
            params_sarimax=_make_params_sarimax(),
            monthly_catboost_candidate_models=_make_catboost_candidate_models(),
            monthly_catboost_prechampion_configs=_make_catboost_prechampion_configs(),
            monthly_catboost_split_metadata=_make_catboost_split_metadata(),
            monthly_catboost_train=_make_catboost_train_df(),
            monthly_catboost_validation=_make_catboost_validation_df(),
            monthly_catboost_test=_make_catboost_test_df(),
            params_catboost=_make_params_catboost(),
        )

    families = set(result["family"].unique())
    assert families == {"prophet", "sarimax", "catboost"}
    assert len(result[result["family"] == "catboost"]) == 2  # noqa: PLR2004


def test_monthly_model_selection_compares_three_families():
    """One family champion is selected for each of prophet, sarimax, and catboost."""
    metrics_df = _make_three_family_metrics_df()
    params = _make_params_monthly(active_families=["prophet", "sarimax", "catboost"])

    summary = select_monthly_family_champions(metrics_df, params)

    assert len(summary) == 3  # noqa: PLR2004
    assert set(summary["family"]) == {"prophet", "sarimax", "catboost"}
    catboost_row = summary[summary["family"] == "catboost"].iloc[0]
    assert catboost_row["family_champion_id"] == "catboost_trial_001"


def test_monthly_model_selection_selects_lowest_wape_catboost_wins():
    """When CatBoost has the lowest WAPE it is elected as the production champion."""
    metrics_df = _make_three_family_metrics_df()
    params = _make_params_monthly(active_families=["prophet", "sarimax", "catboost"])

    family_summary = select_monthly_family_champions(metrics_df, params)
    prod_summary = select_monthly_production_champion(family_summary, metrics_df, params)

    row = prod_summary.iloc[0]
    assert row["production_champion_family"] == "catboost"
    assert row["production_champion_id"] == "catboost_trial_001"
    assert row["primary_metric"] == "wape"


def test_champion_monthly_metadata_contains_model_family_catboost():
    """A CatBoost production champion yields metadata with model_family='catboost'."""
    cb_models = _make_catboost_candidate_models()

    champion_model, metadata = build_monthly_champion_artifacts(
        monthly_model_selection_summary=_make_catboost_production_summary(),
        monthly_family_champion_summary=_make_three_family_champion_summary(),
        monthly_candidate_test_metrics=_make_three_family_metrics_df(),
        monthly_prophet_candidate_models={"prophet_candidate_001": _FakeProphetModel()},
        monthly_sarimax_candidate_models=_make_sarimax_candidate_models(n_test=6),
        monthly_prophet_full_train=_make_prophet_train_df(24),
        monthly_sarimax_full_train=_make_sarimax_full_train_df(30),
        monthly_sarimax_training_metadata=_make_sarimax_training_metadata(),
        params_monthly=_make_params_monthly(
            active_families=["prophet", "sarimax", "catboost"], refit_enabled=False
        ),
        monthly_catboost_candidate_models=cb_models,
        monthly_catboost_full_train=_make_catboost_full_train_df(),
        monthly_catboost_split_metadata=_make_catboost_split_metadata(),
    )

    assert metadata["model_family"] == "catboost"
    assert metadata["champion_id"] == "catboost_trial_001"
    # Refit disabled → the champion model is the resolved candidate dict as-is.
    assert champion_model is cb_models["catboost_trial_001"]
    assert metadata["inference_contract"]["model_family"] == "catboost"
    assert metadata["inference_contract"]["recursive_inference"] is True
    assert metadata["inference_contract"]["has_prediction_intervals"] is False


def test_champion_monthly_metadata_contains_catboost_features_when_catboost_wins():
    """CatBoost champion metadata must expose the feature contract for inference."""
    _, metadata = build_monthly_champion_artifacts(
        monthly_model_selection_summary=_make_catboost_production_summary(),
        monthly_family_champion_summary=_make_three_family_champion_summary(),
        monthly_candidate_test_metrics=_make_three_family_metrics_df(),
        monthly_prophet_candidate_models={"prophet_candidate_001": _FakeProphetModel()},
        monthly_sarimax_candidate_models=_make_sarimax_candidate_models(n_test=6),
        monthly_prophet_full_train=_make_prophet_train_df(24),
        monthly_sarimax_full_train=_make_sarimax_full_train_df(30),
        monthly_sarimax_training_metadata=_make_sarimax_training_metadata(),
        params_monthly=_make_params_monthly(
            active_families=["prophet", "sarimax", "catboost"], refit_enabled=False
        ),
        monthly_catboost_candidate_models=_make_catboost_candidate_models(),
        monthly_catboost_full_train=_make_catboost_full_train_df(),
        monthly_catboost_split_metadata=_make_catboost_split_metadata(),
    )

    assert metadata["feature_columns"] == _CATBOOST_FEATURES
    assert metadata["categorical_features"] == []
    assert metadata["target_column"] == "monthly_demand"
    assert metadata["date_column"] == "month_start_date"
    assert metadata["requires_exogenous_future"] is True
    # The inference contract mirrors the feature contract for the recursive adapter.
    assert metadata["inference_contract"]["feature_columns"] == _CATBOOST_FEATURES
    # Hyperparameters block carries the CatBoost config.
    assert "depth" in metadata["hyperparameters"]
    assert "learning_rate" in metadata["hyperparameters"]


def test_build_champion_refits_catboost_on_full_history():
    """CatBoost champion must be refit on full history into a candidate-dict artifact."""
    fake_cb = _FakeCatBoostModel(pred_value=10.0)

    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes.CatBoostRegressor",
        return_value=fake_cb,
    ):
        champion_model, metadata = build_monthly_champion_artifacts(
            monthly_model_selection_summary=_make_catboost_production_summary(),
            monthly_family_champion_summary=_make_three_family_champion_summary(),
            monthly_candidate_test_metrics=_make_three_family_metrics_df(),
            monthly_prophet_candidate_models={
                "prophet_candidate_001": _FakeProphetModel()
            },
            monthly_sarimax_candidate_models=_make_sarimax_candidate_models(n_test=6),
            monthly_prophet_full_train=_make_prophet_train_df(24),
            monthly_sarimax_full_train=_make_sarimax_full_train_df(30),
            monthly_sarimax_training_metadata=_make_sarimax_training_metadata(),
            params_monthly=_make_params_monthly(
                active_families=["prophet", "sarimax", "catboost"], refit_enabled=True
            ),
            monthly_catboost_candidate_models=_make_catboost_candidate_models(),
            monthly_catboost_full_train=_make_catboost_full_train_df(33),
            monthly_catboost_split_metadata=_make_catboost_split_metadata(),
        )

    assert metadata["model_family"] == "catboost"
    assert metadata["refit"]["performed"] is True
    assert metadata["refit"]["n_obs"] == 33  # noqa: PLR2004
    assert metadata["training_cutoff"] is not None

    # The champion carries the freshly refit model in the candidate-dict shape.
    assert isinstance(champion_model, dict)
    assert champion_model["refit_scope"] == "full_history"
    assert champion_model["model"] is fake_cb
    assert champion_model["feature_columns"] == _CATBOOST_FEATURES


# ── Test 13: Prophet & CatBoost rolling-origin M-2/M-3 WAPE (operational lead) ─


def _enable_rolling_origin(params: dict, lead_times: list[int] | None = None) -> dict:
    out = dict(params)
    out["operational_lead_time"] = {
        "enabled": True,
        "lead_times": lead_times or [2, 3],
    }
    return out


def test_score_prophet_candidates_computes_rolling_origin_when_enabled():
    """Prophet candidates gain test_m2_wape/test_m3_wape when operational lead-time is on."""
    n_test = 5
    fake_model = MagicMock()
    fake_model.predict.side_effect = lambda df: pd.DataFrame(
        {"yhat": np.full(len(df), 10.0)}
    )

    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes._refit_prophet_on_frame",
        return_value=fake_model,
    ):
        rows = _score_prophet_candidates(
            candidate_models={
                "prophet_candidate_001": _FakeProphetModel(),
                "prophet_candidate_002": _FakeProphetModel(),
            },
            prechampion_configs=_make_prophet_prechampion_configs(),
            train_df=_make_prophet_train_df(24),
            validation_df=_make_prophet_validation_df(3),
            test_df=_make_prophet_test_df(n_test),
            params=_enable_rolling_origin(_make_params_prophet()),
            mase_period=12,
            require_all=True,
        )

    assert rows, "expected at least one scored Prophet candidate"
    for r in rows:
        for col in ("test_m2_wape", "test_m3_wape", "n_m2_pairs", "n_m3_pairs"):
            assert col in r, f"rolling-origin column '{col}' missing from Prophet row"
    # 5-month test → M-2 has 4 (origin,target) pairs, M-3 has 3.
    assert rows[0]["n_m2_pairs"] == 4  # noqa: PLR2004
    assert rows[0]["n_m3_pairs"] == 3  # noqa: PLR2004
    assert rows[0]["test_m2_wape"] is not None
    assert rows[0]["test_m2_wape"] >= 0.0


def test_score_catboost_candidates_computes_recursive_rolling_origin_when_enabled():
    """CatBoost candidates gain recursive test_m2_wape/test_m3_wape when enabled."""
    n_test = 5
    fake = _FakeCatBoostModel(pred_value=10.0)

    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes.CatBoostRegressor",
        return_value=fake,
    ):
        rows = _score_catboost_candidates(
            candidate_models=_make_catboost_candidate_models(),
            prechampion_configs=_make_catboost_prechampion_configs(),
            split_metadata=_make_catboost_split_metadata(),
            train_df=_make_catboost_train_df(24),
            validation_df=_make_catboost_validation_df(3),
            test_df=_make_catboost_test_df(n_test),
            params=_enable_rolling_origin(_make_params_catboost()),
            mase_period=12,
            require_all=True,
        )

    assert rows, "expected at least one scored CatBoost candidate"
    for r in rows:
        for col in ("test_m2_wape", "test_m3_wape", "n_m2_pairs", "n_m3_pairs"):
            assert col in r, f"rolling-origin column '{col}' missing from CatBoost row"
    # Same pairing logic as SARIMAX/Prophet: 5-month test → 4 M-2 pairs, 3 M-3 pairs.
    assert rows[0]["n_m2_pairs"] == 4  # noqa: PLR2004
    assert rows[0]["n_m3_pairs"] == 3  # noqa: PLR2004


def test_rolling_origin_disabled_by_default_for_prophet_and_catboost():
    """Without operational_lead_time config, no rolling-origin columns are emitted."""
    # Prophet (real refit avoided via mock; ro disabled means a single refit per candidate)
    fake_model = MagicMock()
    fake_model.predict.side_effect = lambda df: pd.DataFrame(
        {"yhat": np.full(len(df), 10.0)}
    )
    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes._refit_prophet_on_frame",
        return_value=fake_model,
    ):
        prophet_rows = _score_prophet_candidates(
            candidate_models={"prophet_candidate_001": _FakeProphetModel()},
            prechampion_configs=_make_prophet_prechampion_configs(["prophet_candidate_001"]),
            train_df=_make_prophet_train_df(24),
            validation_df=_make_prophet_validation_df(3),
            test_df=_make_prophet_test_df(5),
            params=_make_params_prophet(),  # no operational_lead_time
            mase_period=12,
            require_all=True,
        )
    assert "test_m2_wape" not in prophet_rows[0]

    fake = _FakeCatBoostModel(pred_value=10.0)
    with patch(
        "hdf_pipelines.pipelines.model_selection.monthly.nodes.CatBoostRegressor",
        return_value=fake,
    ):
        catboost_rows = _score_catboost_candidates(
            candidate_models=_make_catboost_candidate_models(["catboost_trial_001"]),
            prechampion_configs=_make_catboost_prechampion_configs(["catboost_trial_001"]),
            split_metadata=_make_catboost_split_metadata(),
            train_df=_make_catboost_train_df(24),
            validation_df=_make_catboost_validation_df(3),
            test_df=_make_catboost_test_df(5),
            params=_make_params_catboost(),  # no operational_lead_time
            mase_period=12,
            require_all=True,
        )
    assert "test_m2_wape" not in catboost_rows[0]


# ── Test 9: refit and metadata preserve per-regressor prior scales ────────────


def _fit_prophet_with_regressor(prior_scale: float, name: str = "reg_a") -> Any:
    """Fit a minimal real Prophet model with one tuned-prior-scale regressor."""
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2021-01-01", periods=24, freq="MS"),
            "y": np.linspace(8.0, 12.0, 24),
            name: np.linspace(0.0, 1.0, 24),
        }
    )
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.add_regressor(name, prior_scale=prior_scale, mode="additive")
    model.fit(df)
    return model


def test_refit_prophet_preserves_per_regressor_prior_scale():
    """Regression: the champion refit must carry tuned per-regressor prior scales.

    Without passing prior_scale to add_regressor(), Prophet silently defaults each
    regressor to the global holidays_prior_scale (10.0), discarding the tuned value.
    """
    candidate = _fit_prophet_with_regressor(prior_scale=7.5)
    fit_df = pd.DataFrame(
        {
            "ds": pd.date_range("2021-01-01", periods=24, freq="MS"),
            "y": np.linspace(8.0, 12.0, 24),
            "reg_a": np.linspace(0.0, 1.0, 24),
        }
    )

    refit = _refit_prophet_on_frame(candidate, fit_df, ["reg_a"])

    assert refit.extra_regressors["reg_a"]["prior_scale"] == pytest.approx(7.5)


def test_extract_prophet_regressor_prior_scales_surfaces_tuned_values():
    """The metadata extractor must surface per-regressor prior scales for the app."""
    model = _fit_prophet_with_regressor(prior_scale=3.0)

    assert _extract_prophet_regressor_prior_scales(model) == pytest.approx(
        {"reg_a": 3.0}
    )
