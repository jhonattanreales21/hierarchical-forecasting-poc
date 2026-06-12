"""Tests for monthly CatBoost training (rolling-origin, direct multi-horizon E1).

Covers:
- train_monthly_catboost_candidates: 5-output node with rolling-origin, direct E1.
- Per-horizon model training: 3 independent models stored per prechampion.
- prechampion_configs format: rolling_origin_metrics present per candidate.
- _make_direct_fit_forecast_fn and _refit_direct_models_on_df helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hdf_pipelines.pipelines.train_monthly.catboost.nodes import (
    _make_direct_fit_forecast_fn,
    _refit_direct_models_on_df,
    train_monthly_catboost_candidates,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_demand_df(n_months: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2021-01-01", periods=n_months, freq="MS")
    demand = rng.integers(100, 500, size=n_months).astype(float)
    df = pd.DataFrame(
        {"month_start_date": dates, "monthly_demand": demand, "sku": "SKU_T"}
    )
    df["month"] = df["month_start_date"].dt.month.astype("int64")
    df["demand_lag_1"] = df["monthly_demand"].shift(1)
    df["demand_lag_2"] = df["monthly_demand"].shift(2)
    df["demand_lag_3"] = df["monthly_demand"].shift(3)
    df["rolling_mean_3"] = df["monthly_demand"].shift(1).rolling(3).mean()
    return df.dropna().reset_index(drop=True)


@pytest.fixture
def full_train_df() -> pd.DataFrame:
    return _make_demand_df(n_months=40)


@pytest.fixture
def split_metadata() -> dict:
    return {
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "sku_column": "sku",
        "all_feature_columns": [
            "month",
            "demand_lag_1",
            "demand_lag_2",
            "demand_lag_3",
            "rolling_mean_3",
        ],
        "direct_multi_horizon": {"strategy": "direct", "horizons": [1, 2, 3]},
    }


@pytest.fixture
def params() -> dict:
    return {
        "primary_metric": "wmape_m3",
        "selection_direction": "minimize",
        "random_seed": 42,
        "prechampion_count": 2,
        "mase_seasonal_period": 12,
        "epsilon": 1.0,
        "max_trials": 3,
        "max_failed_trials": 5,
        "sampler": {
            "name": "tpe",
            "seed": 42,
            "n_startup_trials": 3,
            "multivariate": False,
            "gamma": 0.4,
        },
        "pruning": {"enabled": False},
        "rolling_origin": {
            "horizon": 3,
            "n_cycles": 3,
            "window": "expanding",
            "step_months": 1,
            "min_train_periods": 6,
            "mase_seasonal_period": 12,
        },
        "search_space": {
            "depth": {"type": "int", "low": 2, "high": 3},
            "iterations": {"type": "int", "low": 5, "high": 10},
            "learning_rate": {"type": "float", "low": 0.1, "high": 0.3, "log": False},
        },
        "fixed_params": {"loss_function": "RMSE"},
    }


# ── Node tests ────────────────────────────────────────────────────────────────


def test_node_returns_five_outputs(full_train_df, split_metadata, params):
    result = train_monthly_catboost_candidates(full_train_df, split_metadata, params)
    assert isinstance(result, tuple) and len(result) == 5
    tuning_df, ro_df, pc_configs, cand_models, metadata = result
    assert isinstance(tuning_df, pd.DataFrame)
    assert isinstance(ro_df, pd.DataFrame)
    assert isinstance(pc_configs, dict)
    assert isinstance(cand_models, dict)
    assert isinstance(metadata, dict)


def test_tuning_df_has_rank_and_ro_metrics(full_train_df, split_metadata, params):
    tuning_df, *_ = train_monthly_catboost_candidates(
        full_train_df, split_metadata, params
    )
    assert "rank" in tuning_df.columns
    assert (tuning_df["rank"] == 1).any()
    assert "ro_wmape_m3" in tuning_df.columns
    assert "ro_wmape" in tuning_df.columns


def test_prechampion_configs_rolling_origin_format(
    full_train_df, split_metadata, params
):
    _, _, pc_configs, _, _ = train_monthly_catboost_candidates(
        full_train_df, split_metadata, params
    )
    assert pc_configs["model_family"] == "catboost"
    assert pc_configs["selection_stage"] == "rolling_origin"
    candidates = pc_configs["candidates"]
    assert len(candidates) > 0
    for cand in candidates:
        assert "rolling_origin_metrics" in cand
        assert "wmape_m3" in cand["rolling_origin_metrics"]
        assert "feature_columns" in cand
        assert "config" in cand


def test_candidate_models_have_three_direct_models(
    full_train_df, split_metadata, params
):
    _, _, _, cand_models, _ = train_monthly_catboost_candidates(
        full_train_df, split_metadata, params
    )
    assert len(cand_models) > 0
    for cid, entry in cand_models.items():
        assert "model_h1" in entry, f"{cid}: missing model_h1"
        assert "model_h2" in entry, f"{cid}: missing model_h2"
        assert "model_h3" in entry, f"{cid}: missing model_h3"
        assert entry.get("strategy") == "direct_multi_horizon"
        assert "feature_columns" in entry


def test_training_metadata_fields(full_train_df, split_metadata, params):
    _, _, _, _, metadata = train_monthly_catboost_candidates(
        full_train_df, split_metadata, params
    )
    assert metadata["evaluation_mode"] == "rolling_origin"
    assert metadata["strategy"] == "direct_multi_horizon_e1"
    assert metadata["model_family"] == "catboost"
    assert "n_cycles" in metadata
    assert "horizon" in metadata


def test_raises_on_empty_full_train(split_metadata, params):
    with pytest.raises(ValueError, match="empty"):
        train_monthly_catboost_candidates(pd.DataFrame(), split_metadata, params)


def test_raises_on_missing_required_column(full_train_df, params):
    bad_meta = {
        "date_column": "month_start_date",
        "target_column": "nonexistent",
        "sku_column": "sku",
        "all_feature_columns": ["month"],
    }
    with pytest.raises(ValueError):
        train_monthly_catboost_candidates(full_train_df, bad_meta, params)


# ── Helper tests ──────────────────────────────────────────────────────────────


def test_make_direct_fit_forecast_fn_returns_three_predictions(full_train_df):
    config = {
        "depth": 2,
        "iterations": 5,
        "learning_rate": 0.1,
        "loss_function": "RMSE",
        "random_seed": 42,
    }
    feature_cols = ["month", "demand_lag_1", "demand_lag_2"]
    fit_fn = _make_direct_fit_forecast_fn(
        config=config,
        feature_cols=feature_cols,
        target_col="monthly_demand",
        sku_col="sku",
        horizons=[1, 2, 3],
    )
    train_subset = full_train_df.iloc[:20].copy()
    # The cycle arg is unused in the direct E1 implementation (only train_df matters).
    from unittest.mock import MagicMock

    cycle = MagicMock()
    preds = fit_fn(train_subset, cycle)
    assert len(preds) == 3
    assert all(np.isfinite(p) for p in preds)


def test_refit_direct_models_on_df_returns_three_models(full_train_df):
    from catboost import CatBoostRegressor

    config = {
        "depth": 2,
        "iterations": 5,
        "learning_rate": 0.1,
        "loss_function": "RMSE",
        "random_seed": 42,
    }
    feature_cols = ["month", "demand_lag_1", "demand_lag_2"]
    result = _refit_direct_models_on_df(
        config=config,
        df=full_train_df,
        feature_cols=feature_cols,
        target_col="monthly_demand",
        sku_col="sku",
        horizons=[1, 2, 3],
    )
    for h in [1, 2, 3]:
        assert f"model_h{h}" in result
        m = result[f"model_h{h}"]
        assert isinstance(m, CatBoostRegressor)
        X = full_train_df[feature_cols].iloc[[-1]].to_numpy(dtype=float)
        assert np.isfinite(m.predict(X)[0])
