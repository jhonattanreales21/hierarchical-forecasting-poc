"""Tests for monthly Prophet rolling-origin training nodes."""

from __future__ import annotations

from typing import Any

import pandas as pd

from hdf_pipelines.pipelines.train_monthly.prophet import nodes


class _FakeProphetModel:
    """Minimal Prophet-like test double with deterministic predictions."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def fit(self, train_df: pd.DataFrame) -> "_FakeProphetModel":
        if self.config["seasonality_mode"] == "broken":
            raise ValueError("synthetic trial failure")
        return self

    def predict(self, predict_df: pd.DataFrame) -> pd.DataFrame:
        base = 10.0 if self.config["seasonality_mode"] == "additive" else 12.0
        yhat = [base + 2.0 * i for i in range(len(predict_df))]
        return pd.DataFrame({"ds": predict_df["ds"].values, "yhat": yhat})


def _build_full_history(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=n, freq="MS"),
            "y": [float(8 + i) for i in range(n)],
            "sku": ["SKU-1"] * n,
            "regressor_1": [1.0] * n,
        }
    )


def _build_params() -> dict[str, Any]:
    return {
        "date_column": "ds",
        "target_column": "y",
        "sku_column": "sku",
        "active_regressors": ["regressor_1"],
        "regressors": {"mode": "additive"},
        "tuning": {
            "optimizer": "optuna",
            "objective": {"metric": "wmape_m3", "direction": "minimize"},
            "max_trials": 3,
            "top_n_prechampions": 2,
            "sampler": {"name": "tpe", "seed": 42},
            "rolling_origin": {
                "horizon": 3,
                "n_cycles": 3,
                "window": "expanding",
                "step_months": 1,
                "min_train_periods": 10,
                "mase_seasonal_period": 12,
            },
            "pruning": {"enabled": False},
            "search_space": {
                "changepoint_prior_scale": {"type": "float", "low": 0.01, "high": 0.5},
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
        "metrics": {"epsilon": 1.0, "mase_seasonal_period": 12},
    }


EXPECTED_SUCCESSFUL_TRIALS = 2
EXPECTED_TOTAL_TRIALS = 3


def _patch(monkeypatch):
    def fake_suggest_trial_params(trial, search_space, fixed_params):
        modes = ["additive", "multiplicative", "broken"]
        config = {"changepoint_prior_scale": 0.05, "seasonality_mode": modes[trial.number]}
        config.update(fixed_params)
        return config

    monkeypatch.setattr(nodes, "suggest_trial_params", fake_suggest_trial_params)
    monkeypatch.setattr(
        nodes,
        "_create_prophet_model",
        lambda config, active_regressors, regressor_mode: _FakeProphetModel(config),
    )


def test_prophet_rolling_origin_ranks_and_emits_wmape_m3(monkeypatch):
    _patch(monkeypatch)
    full_train = _build_full_history()
    split_metadata = {"active_regressors": ["regressor_1"]}

    (
        tuning_results,
        ro_metrics,
        prechampion_configs,
        candidate_models,
        training_metadata,
        best_model,
    ) = nodes.train_and_evaluate_monthly_prophet_candidates(
        full_train, split_metadata, _build_params()
    )

    # Two trials succeed, the "broken" one fails (all cycles raise).
    statuses = dict(zip(tuning_results["candidate_id"], tuning_results["status"], strict=True))
    assert statuses["prophet_candidate_003"] == "failed"
    assert sum(s == "success" for s in statuses.values()) == EXPECTED_SUCCESSFUL_TRIALS

    # Objective + persisted metric set are rolling-origin.
    assert tuning_results["selection_metric"].iloc[0] == "wmape_m3"
    assert "wmape_m3" in ro_metrics.columns
    assert training_metadata["evaluation_mode"] == "rolling_origin"
    assert training_metadata["rolling_origin"]["horizon"] == 3

    # Pre-champions are ranked; candidate models are refit on full history.
    prechampion_ids = [c["candidate_id"] for c in prechampion_configs["prechampions"]]
    assert len(prechampion_ids) == 2
    assert best_model is candidate_models[prechampion_ids[0]]
    assert "rolling_origin_metrics" in prechampion_configs["prechampions"][0]


def test_prophet_prechampion_contract(monkeypatch):
    _patch(monkeypatch)
    full_train = _build_full_history()
    params = _build_params()
    params["tuning"]["max_trials"] = 2  # only the two succeeding configs

    tuning_results, _, prechampion_configs, _, _, _ = (
        nodes.train_and_evaluate_monthly_prophet_candidates(
            full_train, {"active_regressors": ["regressor_1"]}, params
        )
    )

    required_columns = {
        "candidate_id", "rank", "is_prechampion",
        "selection_metric", "selection_metric_value",
    }
    assert required_columns.issubset(tuning_results.columns)

    first = prechampion_configs["prechampions"][0]
    assert first["rank"] == 1
    assert first["active_regressors"] == ["regressor_1"]
    assert {
        "changepoint_prior_scale", "seasonality_prior_scale", "holidays_prior_scale",
        "seasonality_mode", "yearly_seasonality", "weekly_seasonality",
        "daily_seasonality", "interval_width",
    }.issubset(first["model_params"])
