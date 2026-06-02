"""Tests for monthly CatBoost training nodes (Optuna TPE-based tuning)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hdf_pipelines.pipelines.train_monthly.catboost.nodes import (
    _compute_validation_metrics,
    _rank_candidates,
    train_monthly_catboost_candidates,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

_N_TRAIN = 24
_N_VAL = 3


def _build_train_val_inputs(
    n_train: int = _N_TRAIN,
    n_val: int = _N_VAL,
    include_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic monthly train / validation DataFrames with CatBoost features."""
    demand_train = np.linspace(50.0, 150.0, n_train)
    dates_train = pd.date_range("2021-01-01", periods=n_train, freq="MS")

    demand_val = [155.0, 160.0, 165.0][:n_val]
    dates_val = pd.date_range(
        dates_train[-1] + pd.DateOffset(months=1), periods=n_val, freq="MS"
    )

    base: dict[str, Any] = {
        "month_start_date": dates_train,
        "monthly_demand": demand_train,
        "sku": ["SKU-1"] * n_train,
    }
    base_val: dict[str, Any] = {
        "month_start_date": dates_val,
        "monthly_demand": demand_val,
        "sku": ["SKU-1"] * n_val,
    }

    if include_features:
        base["month"] = [d.month for d in dates_train]
        base["demand_lag_1"] = [np.nan] + list(demand_train[:-1])
        base["demand_lag_2"] = [np.nan, np.nan] + list(demand_train[:-2])
        base["rolling_mean_3"] = pd.Series(demand_train).shift(1).rolling(3).mean().tolist()
        base["business_days"] = [20] * n_train

        val_all = list(demand_train) + demand_val
        base_val["month"] = [d.month for d in dates_val]
        base_val["demand_lag_1"] = [val_all[n_train - 1 + i] for i in range(n_val)]
        base_val["demand_lag_2"] = [val_all[n_train - 2 + i] for i in range(n_val)]
        base_val["rolling_mean_3"] = [
            np.mean(val_all[n_train - 3 + i : n_train + i]) for i in range(n_val)
        ]
        base_val["business_days"] = [20] * n_val

    train = pd.DataFrame(base)
    val = pd.DataFrame(base_val)

    if include_features:
        train = train.dropna(subset=["demand_lag_1", "demand_lag_2"]).reset_index(drop=True)

    return train, val


def _build_split_metadata(feature_columns: list[str] | None = None) -> dict:
    """Build a minimal CatBoost split metadata dict."""
    return {
        "model_family": "catboost",
        "granularity": "monthly",
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "sku_column": "sku",
        "all_feature_columns": feature_columns
        or ["month", "demand_lag_1", "demand_lag_2", "rolling_mean_3", "business_days"],
    }


def _build_params(
    prechampion_count: int = 2,
    random_seed: int = 42,
    max_trials: int = 2,
) -> dict[str, Any]:
    """Build a minimal params dict for the CatBoost training node (Optuna format)."""
    return {
        "primary_metric": "wape",
        "selection_direction": "minimize",
        "random_seed": random_seed,
        "early_stopping_rounds": 10,
        "prechampion_count": prechampion_count,
        "mase_seasonal_period": 12,
        "epsilon": 1.0,
        "max_trials": max_trials,
        "max_failed_trials": 5,
        "sampler": {"name": "tpe", "seed": random_seed},
        "search_space": {
            "depth": {"type": "int", "low": 4, "high": 6},
            "learning_rate": {"type": "float", "low": 0.05, "high": 0.1},
            "iterations": {"type": "int", "low": 50, "high": 100},
        },
        "fixed_params": {"loss_function": "RMSE"},
    }


def _make_fake_catboost_class(
    predictions: list[float] | None = None,
    fail: bool = False,
) -> type:
    """Return a CatBoostRegressor class mock that produces deterministic predictions."""
    preds = predictions or [155.0, 160.0, 165.0]

    class FakeModel:
        def __init__(self, **kwargs):
            if fail:
                raise ValueError("synthetic CatBoost fit failure")
            self._kwargs = kwargs

        def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=None):
            return self

        def predict(self, X):
            return np.array(preds[: len(X)])

    return FakeModel


# ── test: all artifacts returned ──────────────────────────────────────────────


def test_training_returns_all_artifacts():
    """All five output artifacts must be returned when at least one trial succeeds."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params()
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        result = train_monthly_catboost_candidates(train, val, metadata, params)

    assert len(result) == 5
    tuning_results, val_metrics, prechampion_configs, candidate_models, training_metadata = result

    assert isinstance(tuning_results, pd.DataFrame)
    assert isinstance(val_metrics, pd.DataFrame)
    assert isinstance(prechampion_configs, dict)
    assert isinstance(candidate_models, dict)
    assert isinstance(training_metadata, dict)


# ── test: feature columns exclude identity columns ────────────────────────────


def test_feature_columns_exclude_identity_cols():
    """date, target, and sku columns must not appear in the feature matrix."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params()
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        _, _, _, _, training_metadata = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    feature_columns = training_metadata["feature_columns"]
    assert "month_start_date" not in feature_columns
    assert "monthly_demand" not in feature_columns
    assert "sku" not in feature_columns


# ── test: candidate IDs are stable and unique ─────────────────────────────────


def test_candidate_ids_are_stable_and_unique():
    """Each trial must have a unique, deterministic candidate_id with the expected prefix."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params(max_trials=2)
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        tuning_results, _, _, _, _ = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    ids = tuning_results["candidate_id"].tolist()
    assert len(ids) == len(set(ids)), "Candidate IDs must be unique."
    assert all(cid.startswith("catboost_trial_") for cid in ids)


# ── test: validation metrics contain required keys ────────────────────────────


def test_validation_metrics_contain_required_keys():
    """Each successful trial must expose wape, mase, rmse, and bias in the metrics table."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params()
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        _, val_metrics, _, _, _ = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    assert len(val_metrics) > 0
    for required_col in ("wape", "mase", "rmse", "bias"):
        assert required_col in val_metrics.columns, (
            f"Required metric column {required_col!r} missing from validation_metrics."
        )


# ── test: prechampion configs count ──────────────────────────────────────────


def test_prechampion_configs_count():
    """The prechampion_configs artifact must contain at most prechampion_count entries."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params(prechampion_count=2, max_trials=3)
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        _, _, prechampion_configs, _, _ = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    assert prechampion_configs["model_family"] == "catboost"
    assert prechampion_configs["granularity"] == "monthly"
    assert len(prechampion_configs["candidates"]) <= 2


# ── test: candidate_models keyed by candidate_id ─────────────────────────────


def test_candidate_models_keyed_by_candidate_id():
    """candidate_models dict must be keyed by valid candidate_id strings."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params(prechampion_count=2)
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        tuning_results, _, _, candidate_models, _ = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    assert len(candidate_models) > 0
    all_ids = tuning_results["candidate_id"].tolist()
    for cid in candidate_models:
        assert cid in all_ids, f"candidate_models key {cid!r} not found in tuning_results."
        assert "model" in candidate_models[cid]
        assert "validation_metrics" in candidate_models[cid]
        assert "feature_columns" in candidate_models[cid]


# ── test: training_metadata contains required fields ─────────────────────────


def test_training_metadata_contains_required_fields():
    """training_metadata must document model_family, granularity, run details, and metrics."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params()
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        _, _, _, _, training_metadata = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    required_fields = [
        "model_family",
        "granularity",
        "run_timestamp",
        "optimizer",
        "primary_metric",
        "selection_direction",
        "prechampion_count",
        "n_trials_configured",
        "n_trials_run",
        "n_candidates_successful",
        "n_candidates_failed",
        "best_candidate_id",
        "best_validation_metric",
        "feature_columns",
        "n_features",
        "target_column",
        "date_column",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "search_space",
        "fixed_params",
        "random_seed",
        "selected_prechampion_ids",
        "warnings",
        "failed_candidates",
    ]
    for field in required_fields:
        assert field in training_metadata, (
            f"Required field {field!r} missing from training_metadata."
        )

    assert training_metadata["model_family"] == "catboost"
    assert training_metadata["granularity"] == "monthly"
    assert training_metadata["optimizer"] == "optuna_tpe"


# ── test: failed candidate handled safely ────────────────────────────────────


def test_failed_candidate_handled_safely():
    """One failing trial must not abort the run; remaining trials must succeed."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params(prechampion_count=1, max_trials=2)

    call_count = {"n": 0}
    SuccessModel = _make_fake_catboost_class()

    class FirstFailsThenSucceeds:
        """Fails on the first instantiation, succeeds on subsequent calls."""

        def __new__(cls, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("synthetic first-candidate failure")
            return SuccessModel(**kwargs)

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FirstFailsThenSucceeds,
    ):
        tuning_results, _, _, _, training_metadata = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    assert training_metadata["n_candidates_failed"] >= 1
    assert training_metadata["n_candidates_successful"] >= 1
    assert "failed" in tuning_results["status"].values
    assert "success" in tuning_results["status"].values


# ── test: all candidates fail raises RuntimeError ────────────────────────────


def test_all_candidates_fail_raises_error():
    """RuntimeError must be raised when every configured trial fails to train."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params()
    AlwaysFails = _make_fake_catboost_class(fail=True)

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        AlwaysFails,
    ), pytest.raises(RuntimeError, match="All CatBoost monthly trials failed"):
        train_monthly_catboost_candidates(train, val, metadata, params)


# ── test: feature columns from metadata override DataFrame inference ───────────


def test_feature_columns_from_metadata_are_used():
    """When all_feature_columns is present in metadata, it must be used over DataFrame inference."""
    train, val = _build_train_val_inputs()
    restricted_cols = ["month", "demand_lag_1"]
    metadata = _build_split_metadata(feature_columns=restricted_cols)
    params = _build_params()
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        _, _, _, _, training_metadata = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    assert training_metadata["feature_columns"] == restricted_cols
    assert training_metadata["n_features"] == len(restricted_cols)


# ── test: metadata fields match actual dataset columns ───────────────────────


def test_metadata_feature_columns_match_actual_columns():
    """all_feature_columns in metadata must correspond to real DataFrame columns."""
    train, val = _build_train_val_inputs()
    feature_cols = ["month", "demand_lag_1", "demand_lag_2", "rolling_mean_3", "business_days"]
    metadata = _build_split_metadata(feature_columns=feature_cols)
    params = _build_params()
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        _, _, _, _, training_metadata = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    for col in training_metadata["feature_columns"]:
        assert col in train.columns, (
            f"Feature column {col!r} in metadata but absent from training DataFrame."
        )


# ── unit tests for private helpers ───────────────────────────────────────────


def test_compute_validation_metrics_returns_required_keys():
    """compute_validation_metrics must return wape, mase, rmse, bias, and mae."""
    y_true = np.array([100.0, 110.0, 120.0])
    y_pred = np.array([105.0, 108.0, 122.0])
    y_train = np.linspace(50.0, 100.0, 24)

    metrics = _compute_validation_metrics(y_true, y_pred, y_train, 12, 1.0)

    for key in ("wape", "mase", "rmse", "bias", "mae"):
        assert key in metrics, f"Required metric key {key!r} missing."
    assert isinstance(metrics["wape"], float)
    assert isinstance(metrics["rmse"], float)


def test_rank_candidates_ordering():
    """_rank_candidates must put the best (lowest wape) candidate at rank 1."""
    df = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c3"],
            "status": ["success", "success", "failed"],
            "objective_value": [0.20, 0.10, None],
        }
    )
    ranked = _rank_candidates(df, "wape", "minimize")

    success_ranked = ranked[ranked["status"] == "success"].set_index("candidate_id")
    assert success_ranked.loc["c2", "rank"] == 1
    assert success_ranked.loc["c1", "rank"] == 2
    assert ranked[ranked["status"] == "failed"]["rank"].isna().all()


def test_metadata_optimizer_field_is_optuna_tpe():
    """training_metadata optimizer field must be 'optuna_tpe' (not 'grid_search')."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params()
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        _, _, _, _, training_metadata = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    assert training_metadata["optimizer"] == "optuna_tpe"
    assert "search_space" in training_metadata
    assert "fixed_params" in training_metadata


def test_search_space_keys_recorded_in_metadata():
    """search_space keys from params must appear in training_metadata search_space."""
    train, val = _build_train_val_inputs()
    metadata = _build_split_metadata()
    params = _build_params()
    FakeCat = _make_fake_catboost_class()

    with patch(
        "hdf_pipelines.pipelines.train_monthly.catboost.nodes.CatBoostRegressor",
        FakeCat,
    ):
        _, _, _, _, training_metadata = train_monthly_catboost_candidates(
            train, val, metadata, params
        )

    for key in params["search_space"]:
        assert key in training_metadata["search_space"], (
            f"search_space key {key!r} missing from training_metadata."
        )
