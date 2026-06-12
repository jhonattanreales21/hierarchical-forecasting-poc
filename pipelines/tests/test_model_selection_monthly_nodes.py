# ruff: noqa: PLR2004
"""Tests for monthly rolling-origin model selection nodes."""

from __future__ import annotations

import pandas as pd

from hdf_pipelines.pipelines.model_selection.monthly.nodes import (
    annotate_monthly_candidate_champion_flags,
    assemble_monthly_candidate_metrics,
    build_monthly_champion_artifacts,
    select_monthly_family_champions,
    select_monthly_production_champion,
)


class _FakeProphetModel:
    extra_regressors: dict = {}
    changepoint_prior_scale = 0.05
    seasonality_prior_scale = 10.0
    holidays_prior_scale = 10.0
    seasonality_mode = "additive"
    yearly_seasonality = False
    weekly_seasonality = False
    daily_seasonality = False
    interval_width = 0.8


def _metrics(wmape_m3: float, wmape_m2: float = 0.20, bias: float = 0.01) -> dict:
    return {
        "wmape": 0.12,
        "wmape_m1": 0.10,
        "wmape_m2": wmape_m2,
        "wmape_m3": wmape_m3,
        "mase": 0.80,
        "bias": bias,
        "rmse": 10.0,
    }


def _prophet_prechampions() -> dict:
    return {
        "model_family": "prophet",
        "granularity": "monthly",
        "selection_stage": "rolling_origin",
        "prechampions": [
            {
                "candidate_id": "prophet_candidate_001",
                "rank": 1,
                "active_regressors": [],
                "hyperparams": {"changepoint_prior_scale": 0.05},
                "rolling_origin_metrics": _metrics(0.08),
            },
            {
                "candidate_id": "prophet_candidate_002",
                "rank": 2,
                "active_regressors": [],
                "hyperparams": {"changepoint_prior_scale": 0.20},
                "rolling_origin_metrics": _metrics(0.15),
            },
        ],
    }


def _sarimax_prechampions() -> dict:
    return {
        "model_family": "sarimax",
        "granularity": "monthly",
        "selection_stage": "rolling_origin",
        "candidates": [
            {
                "trial_id": "sarimax_trial_001",
                "rank": 1,
                "order": [1, 0, 0],
                "seasonal_order": [0, 0, 0, 12],
                "trend": None,
                "use_exog": False,
                "rolling_origin_metrics": _metrics(0.05),
                "ljung_box_pvalue": 0.01,
                "autocorrelation_excluded": True,
                "ljung_box_cycle_index": 5,
            },
            {
                "trial_id": "sarimax_trial_002",
                "rank": 2,
                "order": [1, 0, 1],
                "seasonal_order": [0, 0, 0, 12],
                "trend": None,
                "use_exog": False,
                "rolling_origin_metrics": _metrics(0.11, wmape_m2=0.12),
                "ljung_box_pvalue": 0.20,
                "autocorrelation_excluded": False,
                "ljung_box_cycle_index": 5,
            },
        ],
    }


def _params(refit_enabled: bool = False) -> dict:
    return {
        "active_families": ["prophet", "sarimax"],
        "primary_metric": "wmape_m3",
        "direction": "minimize",
        "tie_breakers": ["wmape_m2", "wmape_m1", "mase", "abs_bias"],
        "require_all_active_families": True,
        "mase_seasonal_period": 12,
        "refit_champion": {"enabled": refit_enabled},
    }


def _prophet_full_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "y": [10, 11, 12, 13, 14, 15],
            "sku": ["SKU_001"] * 6,
        }
    )


def _sarimax_full_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "monthly_demand": [10, 11, 12, 13, 14, 15],
        }
    )


def _assembled_metrics() -> pd.DataFrame:
    return assemble_monthly_candidate_metrics(
        _prophet_prechampions(), _sarimax_prechampions(), _params()
    )


def test_assemble_monthly_candidate_metrics_uses_rolling_origin_contract() -> None:
    metrics = _assembled_metrics()

    assert set(metrics["family"]) == {"prophet", "sarimax"}
    assert "wmape_m3" in metrics.columns
    assert "wape" not in metrics.columns
    assert (metrics["selection_stage"] == "rolling_origin").all()
    assert metrics.loc[
        metrics["candidate_id"] == "sarimax_trial_002", "ljung_box_cycle_index"
    ].iloc[0] == 5


def test_family_and_production_champions_rank_on_wmape_m3() -> None:
    metrics = _assembled_metrics()

    family_summary = select_monthly_family_champions(metrics, _params())
    production_summary = select_monthly_production_champion(
        family_summary, metrics, _params()
    )

    champions = dict(
        zip(
            family_summary["family"],
            family_summary["family_champion_id"],
            strict=True,
        )
    )
    assert champions == {
        "prophet": "prophet_candidate_001",
        "sarimax": "sarimax_trial_002",
    }
    row = production_summary.iloc[0]
    assert row["production_champion_family"] == "prophet"
    assert row["production_champion_id"] == "prophet_candidate_001"
    assert row["primary_metric"] == "wmape_m3"


def test_candidate_flags_mark_family_and_production_champions() -> None:
    metrics = _assembled_metrics()
    family_summary = select_monthly_family_champions(metrics, _params())
    production_summary = select_monthly_production_champion(
        family_summary, metrics, _params()
    )

    flagged = annotate_monthly_candidate_champion_flags(
        metrics, family_summary, production_summary
    )

    assert int(flagged["is_family_champion"].sum()) == 2
    assert int(flagged["is_production_champion"].sum()) == 1
    prod = flagged[flagged["is_production_champion"]].iloc[0]
    assert prod["candidate_id"] == "prophet_candidate_001"


def test_build_monthly_champion_artifacts_has_new_metadata_contract() -> None:
    metrics = _assembled_metrics()
    family_summary = select_monthly_family_champions(metrics, _params())
    production_summary = select_monthly_production_champion(
        family_summary, metrics, _params()
    )
    flagged = annotate_monthly_candidate_champion_flags(
        metrics, family_summary, production_summary
    )

    model, metadata = build_monthly_champion_artifacts(
        monthly_model_selection_summary=production_summary,
        monthly_family_champion_summary=family_summary,
        monthly_candidate_metrics=flagged,
        monthly_prophet_candidate_models={"prophet_candidate_001": _FakeProphetModel()},
        monthly_sarimax_candidate_models={},
        monthly_prophet_full_train=_prophet_full_train(),
        monthly_sarimax_full_train=_sarimax_full_train(),
        monthly_sarimax_training_metadata={"exogenous_columns": []},
        params_monthly=_params(refit_enabled=False),
    )

    assert isinstance(model, _FakeProphetModel)
    assert metadata["metrics"]["wmape_m3"] == 0.08
    assert "wape" not in metadata["metrics"]
    assert metadata["evaluation"]["mode"] == "rolling_origin"
    assert "test_period" not in metadata
    assert metadata["training_cutoff"] == "2024-06-01"
    assert metadata["selection"]["primary_metric"] == "wmape_m3"
