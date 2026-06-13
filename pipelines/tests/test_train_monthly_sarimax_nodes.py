# ruff: noqa: PLR2004
"""Tests for monthly SARIMAX rolling-origin training nodes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hdf_pipelines.pipelines.train_monthly.sarimax.nodes import (
    _build_sarimax_config_from_trial,
    _rank_candidates,
    train_and_evaluate_monthly_sarimax_candidates,
)


def _full_history(months: int = 32) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_start_date": pd.date_range("2021-01-01", periods=months, freq="MS"),
            "monthly_demand": np.linspace(100.0, 180.0, months),
        }
    )


def _metadata() -> dict[str, Any]:
    return {
        "model_family": "sarimax",
        "granularity": "monthly",
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "exogenous_columns": [],
    }


def _params(max_trials: int = 1) -> dict[str, Any]:
    single_value = {"type": "categorical", "choices": [0]}
    return {
        "date_column": "month_start_date",
        "target_column": "monthly_demand",
        "objective": {"metric": "wmape_m3", "direction": "minimize"},
        "top_n_prechampions": 1,
        "metrics": {"mase_seasonal_period": 12, "epsilon": 1.0},
        "max_failed_trials": 10,
        "ljung_box": {"enabled": True, "lags": 10, "pvalue_threshold": 0.05},
        "tuning": {
            "optimizer": "optuna",
            "max_trials": max_trials,
            "use_exog": False,
            "rolling_origin": {
                "horizon": 3,
                "n_cycles": 2,
                "window": "expanding",
                "step_months": 1,
                "min_train_periods": 12,
            },
            "pruning": {"enabled": False},
            "sampler": {"name": "tpe", "seed": 42, "n_startup_trials": 1},
            "search_space": {
                "p": single_value,
                "d": single_value,
                "q": single_value,
                "P": single_value,
                "D": single_value,
                "Q": single_value,
                "trend": {"type": "categorical", "choices": ["none"]},
            },
            "fixed_params": {
                "s": 12,
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        },
    }


def _fake_sarimax_class(fit_endog_lengths: list[int]):
    class FakeResult:
        def __init__(self, endog: np.ndarray) -> None:
            self._endog = np.asarray(endog, dtype=float)
            self.resid = np.linspace(0.1, 1.0, len(self._endog))

        def get_forecast(self, steps: int, exog=None):
            out = MagicMock()
            out.predicted_mean = np.repeat(float(self._endog[-1]), steps)
            return out

    class FakeModel:
        def __init__(self, endog, exog=None, **kwargs) -> None:
            self._endog = np.asarray(endog, dtype=float)

        def fit(self, disp: bool = False):
            fit_endog_lengths.append(len(self._endog))
            return FakeResult(self._endog)

    return FakeModel


def test_build_sarimax_config_maps_trend_none() -> None:
    params = {"p": 1, "d": 0, "q": 1, "P": 0, "D": 0, "Q": 0, "trend": "none"}

    config = _build_sarimax_config_from_trial(params, s=12)

    assert config["order"] == [1, 0, 1]
    assert config["seasonal_order"] == [0, 0, 0, 12]
    assert config["trend"] is None


def test_rank_candidates_uses_objective_value() -> None:
    df = pd.DataFrame(
        {
            "trial_id": ["t1", "t2", "t3"],
            "status": ["success", "success", "failed"],
            "objective_value": [0.3, 0.1, None],
        }
    )

    ranked = _rank_candidates(df, "wmape_m3", "minimize")

    assert ranked.loc[ranked["rank"] == 1, "trial_id"].iloc[0] == "t2"
    assert ranked.loc[ranked["trial_id"] == "t3", "rank"].isna().all()


def test_training_emits_rolling_origin_artifacts_and_last_cycle_ljung_box() -> None:
    fit_endog_lengths: list[int] = []
    fake_ljungbox = MagicMock(
        return_value=pd.DataFrame({"lb_pvalue": [0.20]}, index=[10])
    )

    with (
        patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.SARIMAX",
            new=_fake_sarimax_class(fit_endog_lengths),
        ),
        patch(
            "hdf_pipelines.pipelines.train_monthly.sarimax.nodes.acorr_ljungbox",
            new=fake_ljungbox,
        ),
    ):
        (
            tuning_df,
            rolling_metrics,
            prechampions,
            candidate_models,
            metadata,
            rank1,
        ) = train_and_evaluate_monthly_sarimax_candidates(
            _full_history(), _metadata(), _params()
        )

    last_cycle_train_len = len(_full_history()) - 3
    assert fit_endog_lengths.count(last_cycle_train_len) == 2
    assert fit_endog_lengths[-1] == len(_full_history())
    assert fake_ljungbox.call_count == 1

    assert {"wmape", "wmape_m1", "wmape_m2", "wmape_m3", "mase", "bias", "rmse"}.issubset(
        rolling_metrics.columns
    )
    assert rolling_metrics.loc[0, "ljung_box_pvalue"] == pytest.approx(0.20)
    assert not bool(rolling_metrics.loc[0, "autocorrelation_excluded"])
    assert rolling_metrics.loc[0, "ljung_box_cycle_index"] == 2

    candidate = prechampions["candidates"][0]
    assert candidate["rolling_origin_metrics"]["wmape_m3"] == pytest.approx(
        tuning_df.loc[0, "wmape_m3"]
    )
    assert candidate["ljung_box_cycle_index"] == 2
    assert rank1["candidate_id"] in candidate_models
    assert metadata["objective_metric"] == "wmape_m3"
