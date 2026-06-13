"""Regression tests for the unified monthly Prophet regressor contract."""

from __future__ import annotations

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_project_params() -> dict:
    bootstrap_project(_PROJECT_ROOT)
    with KedroSession.create(project_path=_PROJECT_ROOT) as session:
        return session.load_context().params


def test_monthly_prophet_regressor_lists_share_one_contract() -> None:
    params = _load_project_params()

    monthly_regressor_pool = params["model_input_preparation"]["monthly"][
        "active_regressors"
    ]
    prophet_input_regressors = params["model_input_preparation"]["monthly_prophet"][
        "active_regressors"
    ]
    prophet_training_regressors = params["train_monthly"]["prophet"][
        "active_regressors"
    ]

    assert prophet_training_regressors == prophet_input_regressors
    assert set(prophet_input_regressors).issubset(monthly_regressor_pool)
    assert len(prophet_input_regressors) == len(set(prophet_input_regressors))
