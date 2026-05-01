"""Centralised path constants for the Streamlit app.

All pipeline artifact paths are derived from the repository root so the app
works regardless of the working directory it is launched from.
"""

from pathlib import Path

_APP_ROOT = Path(__file__).parents[2]
DATA_ROOT = _APP_ROOT / "pipelines" / "data"

# Champion artifacts (06_models/champions/)
CHAMPION_META = (
    DATA_ROOT / "06_models" / "champions" / "monthly_prophet_champion_metadata.json"
)

# Selection artifacts (06_models/selection/)
SELECTION_FORECAST = (
    DATA_ROOT
    / "06_models"
    / "selection"
    / "monthly_prophet_champion_test_forecast.parquet"
)
SELECTION_SUMMARY = (
    DATA_ROOT
    / "06_models"
    / "selection"
    / "monthly_prophet_model_selection_summary.parquet"
)
TEST_METRICS = (
    DATA_ROOT / "06_models" / "selection" / "monthly_prophet_test_metrics.parquet"
)

# Model input (05_model_input/)
ACTUALS = DATA_ROOT / "05_model_input" / "monthly_prophet_modeling_data.parquet"

# Inference outputs (07_model_output/)
INFERENCE_META = (
    DATA_ROOT / "07_model_output" / "monthly_prophet_inference_metadata.json"
)
FORECAST_LATEST = (
    DATA_ROOT / "07_model_output" / "monthly_prophet_forecast_latest.parquet"
)


def forecast_parquet(horizon_months: int) -> Path:
    """Return the path for a horizon-specific future forecast parquet."""
    return (
        DATA_ROOT
        / "07_model_output"
        / f"monthly_prophet_forecast_{horizon_months}m.parquet"
    )
