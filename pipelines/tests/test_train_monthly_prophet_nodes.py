"""Tests for monthly Prophet Optuna training nodes."""

from __future__ import annotations

from typing import Any

import pandas as pd

from hdf_pipelines.pipelines.train_monthly.prophet import nodes


class _FakeProphetModel:
    """Minimal Prophet-like test double with deterministic predictions."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def fit(self, train_df: pd.DataFrame) -> _FakeProphetModel:
        if self.config["seasonality_mode"] == "broken":
            raise ValueError("synthetic trial failure")
        return self

    def predict(self, predict_df: pd.DataFrame) -> pd.DataFrame:
        if self.config["seasonality_mode"] == "additive":
            yhat = [10.0, 12.0, 14.0]
        else:
            yhat = [12.0, 14.0, 16.0]
        return pd.DataFrame({"ds": predict_df["ds"].values, "yhat": yhat})


def _build_train_validation_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "y": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "sku": ["SKU-1"] * 6,
            "regressor_1": [1.0] * 6,
        }
    )
    validation = pd.DataFrame(
        {
            "ds": pd.date_range("2024-07-01", periods=3, freq="MS"),
            "y": [10.0, 12.0, 14.0],
            "sku": ["SKU-1"] * 3,
            "regressor_1": [1.0] * 3,
        }
    )
    return train, validation


def _build_params() -> dict[str, Any]:
    return {
        "date_column": "ds",
        "target_column": "y",
        "sku_column": "sku",
        "active_regressors": ["regressor_1"],
        "regressors": {"mode": "additive"},
        "tuning": {
            "optimizer": "optuna",
            "objective": {"metric": "mape", "direction": "minimize"},
            "max_trials": 3,
            "top_n_prechampions": 2,
            "sampler": {"name": "tpe", "seed": 42},
            "search_space": {
                "changepoint_prior_scale": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.5,
                },
                "seasonality_mode": {
                    "type": "categorical",
                    "choices": ["additive", "multiplicative"],
                },
            },
            "fixed_params": {
                "seasonality_prior_scale": 1.0,
                "holidays_prior_scale": 1.0,
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "interval_width": 0.8,
            },
        },
        "metrics": {
            "epsilon": 1.0,
            "business_success_precision_threshold": 0.85,
            "horizon_metrics": {"enabled": True, "horizons": [2, 3]},
        },
    }


EXPECTED_SUCCESSFUL_TRIALS = 2
EXPECTED_TOTAL_TRIALS = 3


def test_train_and_evaluate_monthly_prophet_candidates_ranks_trials_and_emits_metadata(
    monkeypatch,
):
    train_df, validation_df = _build_train_validation_inputs()
    params = _build_params()
    split_metadata = {"active_regressors": ["regressor_1"]}

    def fake_suggest_trial_params(trial, search_space, fixed_params):
        candidates = [
            {
                "changepoint_prior_scale": 0.05,
                "seasonality_mode": "additive",
            },
            {
                "changepoint_prior_scale": 0.25,
                "seasonality_mode": "multiplicative",
            },
            {
                "changepoint_prior_scale": 0.50,
                "seasonality_mode": "broken",
            },
        ]
        config = dict(candidates[trial.number])
        config.update(fixed_params)
        return config

    monkeypatch.setattr(nodes, "suggest_trial_params", fake_suggest_trial_params)
    monkeypatch.setattr(
        nodes,
        "_create_prophet_model",
        lambda config, active_regressors, regressor_mode: _FakeProphetModel(config),
    )

    (
        tuning_results,
        validation_metrics,
        prechampion_configs,
        candidate_models,
        training_metadata,
        best_model,
    ) = nodes.train_and_evaluate_monthly_prophet_candidates(
        train_df,
        validation_df,
        split_metadata,
        params,
    )

    assert tuning_results["candidate_id"].tolist() == [
        "prophet_candidate_001",
        "prophet_candidate_002",
        "prophet_candidate_003",
    ]
    assert tuning_results["status"].tolist() == ["success", "success", "failed"]
    assert tuning_results["rank"].tolist()[:2] == [1, 2]
    assert tuning_results["optimizer"].tolist() == ["optuna", "optuna", "optuna"]
    assert tuning_results["objective_direction"].tolist() == [
        "minimize",
        "minimize",
        "minimize",
    ]
    assert tuning_results["selection_metric"].tolist() == ["mape", "mape", "mape"]
    assert training_metadata["best_candidate_id"] == "prophet_candidate_001"
    assert training_metadata["best_trial_number"] == 0
    assert training_metadata["completed_trials"] == EXPECTED_SUCCESSFUL_TRIALS
    assert training_metadata["failed_trials"] == 1
    assert len(training_metadata["trials"]) == EXPECTED_TOTAL_TRIALS
    assert validation_metrics["candidate_id"].tolist() == [
        "prophet_candidate_001",
        "prophet_candidate_002",
    ]
    assert set(candidate_models) == {
        "prophet_candidate_001",
        "prophet_candidate_002",
    }
    assert best_model is candidate_models["prophet_candidate_001"]

    prechampion_ids = [
        item["candidate_id"] for item in prechampion_configs["prechampions"]
    ]
    assert prechampion_ids == ["prophet_candidate_001", "prophet_candidate_002"]


def test_monthly_prophet_optuna_outputs_preserve_stage4_contract(monkeypatch):
    train_df, validation_df = _build_train_validation_inputs()
    params = _build_params()
    split_metadata = {"active_regressors": ["regressor_1"]}

    def fake_suggest_trial_params(trial, search_space, fixed_params):
        config = {
            "changepoint_prior_scale": 0.05 if trial.number == 0 else 0.25,
            "seasonality_mode": "additive" if trial.number == 0 else "multiplicative",
        }
        config.update(fixed_params)
        return config

    monkeypatch.setattr(nodes, "suggest_trial_params", fake_suggest_trial_params)
    monkeypatch.setattr(
        nodes,
        "_create_prophet_model",
        lambda config, active_regressors, regressor_mode: _FakeProphetModel(config),
    )

    tuning_results, _, prechampion_configs, _, _, _ = (
        nodes.train_and_evaluate_monthly_prophet_candidates(
            train_df,
            validation_df,
            split_metadata,
            params | {"tuning": params["tuning"] | {"max_trials": 2}},
        )
    )

    required_columns = {
        "candidate_id",
        "rank",
        "is_prechampion",
        "selection_metric",
        "selection_metric_value",
    }
    assert required_columns.issubset(tuning_results.columns)

    first_prechampion = prechampion_configs["prechampions"][0]
    assert first_prechampion["rank"] == 1
    assert first_prechampion["active_regressors"] == ["regressor_1"]
    assert set(first_prechampion["model_params"]) == {
        "changepoint_prior_scale",
        "seasonality_prior_scale",
        "holidays_prior_scale",
        "seasonality_mode",
        "yearly_seasonality",
        "weekly_seasonality",
        "daily_seasonality",
        "interval_width",
    }
