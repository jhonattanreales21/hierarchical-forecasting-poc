"""Tests for shared Optuna utility helpers."""

import pytest
from optuna.trial import FixedTrial

from hdf_pipelines.utils.optuna_helpers import (
    suggest_trial_params,
    validate_objective_metric_direction,
    validate_optuna_search_space,
)


def test_suggest_trial_params_supports_float_int_and_categorical():
    search_space = validate_optuna_search_space(
        {
            "changepoint_prior_scale": {
                "type": "float",
                "low": 0.01,
                "high": 0.5,
                "log": True,
            },
            "n_changepoints": {
                "type": "int",
                "low": 5,
                "high": 15,
                "step": 5,
            },
            "seasonality_mode": {
                "type": "categorical",
                "choices": ["additive", "multiplicative"],
            },
        }
    )
    trial = FixedTrial(
        {
            "changepoint_prior_scale": 0.05,
            "n_changepoints": 10,
            "seasonality_mode": "multiplicative",
        }
    )

    params = suggest_trial_params(
        trial,
        search_space,
        fixed_params={"yearly_seasonality": True},
    )

    assert params == {
        "changepoint_prior_scale": 0.05,
        "n_changepoints": 10,
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
    }


@pytest.mark.parametrize(
    ("search_space", "expected_message"),
    [
        ({}, "Optuna search_space is empty"),
        (
            {"alpha": {"type": "float", "low": 2.0, "high": 1.0}},
            "low > high",
        ),
        (
            {"depth": {"type": "int", "low": 1, "high": 3, "step": 0}},
            "positive 'step'",
        ),
        (
            {"mode": {"type": "categorical", "choices": []}},
            "non-empty 'choices'",
        ),
    ],
)
def test_validate_optuna_search_space_rejects_invalid_definitions(
    search_space,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        validate_optuna_search_space(search_space)


def test_validate_objective_metric_direction_rejects_incompatible_direction():
    supported_metrics = {"mape", "forecast_precision", "horizon_4_forecast_precision"}

    with pytest.raises(ValueError, match="incompatible"):
        validate_objective_metric_direction("mape", "maximize", supported_metrics)

    metric, direction = validate_objective_metric_direction(
        "horizon_4_forecast_precision",
        "maximize",
        supported_metrics,
    )

    assert (metric, direction) == ("horizon_4_forecast_precision", "maximize")
