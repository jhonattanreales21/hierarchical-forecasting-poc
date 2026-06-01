"""Tests for monthly multi-family model selection nodes (Prophet + SARIMAX).

Covers:
- evaluate_monthly_family_candidates_on_test (Prophet + SARIMAX scoring)
- select_monthly_family_champions (one champion per family)
- select_monthly_production_champion (primary metric + tie-breakers)
- build_monthly_champion_artifacts (metadata schema + model resolution)
- Scope guard: no CatBoost or weekly inputs required
- Hard-fail when active family artifacts are missing
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hdf_pipelines.pipelines.model_selection.monthly.nodes import (
    _score_prophet_candidates,
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


# ── Test 4: champion metadata has required schema ─────────────────────────────


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


# ── Test 5: scope guard — no CatBoost or weekly inputs required ───────────────


def test_monthly_model_selection_does_not_require_catboost_or_weekly_inputs():
    """The monthly multi-family selection nodes accept only prophet+sarimax inputs."""
    import inspect

    from hdf_pipelines.pipelines.model_selection.monthly.nodes import (
        evaluate_monthly_family_candidates_on_test,
    )
    from hdf_pipelines.pipelines.model_selection.monthly.pipeline import create_pipeline

    sig = inspect.signature(evaluate_monthly_family_candidates_on_test)
    param_names = list(sig.parameters.keys())

    for name in param_names:
        assert "catboost" not in name.lower(), (
            f"Parameter '{name}' references CatBoost — not supported in the monthly Prophet+SARIMAX selection pipeline"
        )
        assert "weekly" not in name.lower(), (
            f"Parameter '{name}' references weekly data — not supported in the monthly Prophet+SARIMAX selection pipeline"
        )

    # Pipeline nodes must not reference CatBoost or weekly catalog keys
    p = create_pipeline()
    all_inputs = {inp for n in p.nodes for inp in n.inputs}
    for key in all_inputs:
        assert "catboost" not in key.lower(), (
            f"Pipeline input '{key}' references CatBoost — not supported in the monthly Prophet+SARIMAX selection pipeline"
        )
        assert "weekly" not in key.lower(), (
            f"Pipeline input '{key}' references weekly data — not supported in the monthly Prophet+SARIMAX selection pipeline"
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
