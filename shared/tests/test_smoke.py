"""Minimal smoke tests for the shared package.

These tests only verify that the package imports correctly,
basic public APIs are available, and core schemas can be instantiated.
They are not intended to validate metric correctness exhaustively.
"""

from datetime import date, datetime

import numpy as np


def test_shared_modules_import() -> None:
    # Catches broken installs, missing __init__.py exports, or renamed modules.
    from shared import metrics, schemas

    assert metrics is not None
    assert schemas is not None


def test_metrics_smoke() -> None:
    from shared.metrics import mape, mase, rmse

    # Realistic forecast inputs: 3 periods, predictions slightly off true values.
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    # Training history long enough for seasonality=12 (>12 points required by mase).
    y_train = np.arange(1.0, 25.0)

    # Only checking return type — correctness is tested in dedicated unit tests.
    assert isinstance(mape(y_true, y_pred), float)
    assert isinstance(rmse(y_true, y_pred), float)
    assert isinstance(mase(y_true, y_pred, y_train, seasonality=12), float)


def test_schemas_smoke() -> None:
    from shared.schemas import (
        BacktestResult,
        ForecastArtifact,
        ForecastRecord,
        TemporalGranularity,
    )

    # ForecastRecord: one data point at a known date with a prediction interval.
    record = ForecastRecord(
        ds=date(2024, 1, 1),
        y_actual=100.0,
        y_forecast=105.0,
        y_lower=95.0,
        y_upper=115.0,
        granularity=TemporalGranularity.MONTHLY,
        model_name="prophet",
        horizon_step=1,
    )

    # BacktestResult: aggregated metrics from a rolling-origin backtest run.
    backtest = BacktestResult(
        model_name="prophet",
        granularity=TemporalGranularity.MONTHLY,
        mape=0.10,
        rmse=5.0,
        mase=0.80,
        n_folds=3,
        horizon=1,
    )

    # ForecastArtifact: canonical output written to data/07_model_output/.
    # Bundles the MLflow run ID, forecast records, and backtest evaluation together.
    artifact = ForecastArtifact(
        run_id="smoke-run",
        model_name="prophet",
        granularity=TemporalGranularity.MONTHLY,
        trained_at=datetime(2024, 6, 1, 12, 0, 0),
        forecast_records=[record],
        backtest_result=backtest,
    )

    assert artifact.run_id == "smoke-run"
    assert len(artifact.forecast_records) == 1
    assert artifact.backtest_result.model_name == "prophet"
