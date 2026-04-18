"""Pydantic v2 schemas shared across pipelines, app, and API.

These schemas define the canonical data contracts for forecast artifacts,
evaluation results, and individual forecast records throughout the project.
All inter-package communication involving forecast data should use these types.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class TemporalGranularity(str, Enum):
    """Supported temporal levels in the forecasting hierarchy."""

    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"


class ForecastRecord(BaseModel):
    """Single forecast data point at a given granularity."""

    ds: date                          # period date (month start, week start, or day)
    y_actual: Optional[float]         # actual observed value; None if the period is future
    y_forecast: float                 # point forecast value
    y_lower: Optional[float]          # lower bound of prediction interval
    y_upper: Optional[float]          # upper bound of prediction interval
    granularity: TemporalGranularity  # temporal level this record belongs to
    model_name: str                   # name of the model that produced this forecast
    horizon_step: int                 # 1-based position within the forecast horizon


class BacktestResult(BaseModel):
    """Aggregated evaluation metrics from a time-based backtesting run."""

    model_name: str
    granularity: TemporalGranularity
    mape: float    # Mean Absolute Percentage Error
    rmse: float    # Root Mean Squared Error
    mase: float    # Mean Absolute Scaled Error
    n_folds: int   # number of rolling-origin folds used
    horizon: int   # forecast horizon length (in granularity units)


class ForecastArtifact(BaseModel):
    """Full artifact produced after a training and forecasting run.

    Combines the MLflow tracking reference, the forecast records, and
    the backtesting evaluation in a single serialisable object. This is
    the canonical output written to data/07_model_output/ and consumed
    by the app and API.
    """

    run_id: str                              # MLflow run ID for traceability
    model_name: str
    granularity: TemporalGranularity
    trained_at: datetime                     # UTC timestamp of the training run
    forecast_records: list[ForecastRecord]   # ordered by ds ascending
    backtest_result: BacktestResult
