"""Centralised path constants for the Streamlit app.

All pipeline artifact paths are derived from the repository root so the app
works regardless of the working directory it is launched from.
"""

from pathlib import Path

_APP_ROOT = Path(__file__).parents[2]
DATA_ROOT = _APP_ROOT / "pipelines" / "data"
APP_CACHE_ROOT = _APP_ROOT / "app" / ".cache"
DEFAULT_RAG_DOCUMENT = (
    _APP_ROOT / "pipelines" / "docs" / "recothrom_time_series_history_rag_updated.md"
)


def _first_existing(*paths: Path) -> Path:
    """Return the first existing path, otherwise the first candidate."""
    return next((path for path in paths if path.exists()), paths[0])

# Champion artifacts (06_models/champions/)
CHAMPION_META = _first_existing(
    DATA_ROOT / "06_models" / "champions" / "champion_monthly_metadata.json",
    DATA_ROOT / "06_models" / "champions" / "monthly_prophet_champion_metadata.json",
)

# Selection artifacts (06_models/selection/)
SELECTION_FORECAST = _first_existing(
    DATA_ROOT
    / "06_models"
    / "selection"
    / "monthly_prophet_champion_test_forecast.parquet",
)
SELECTION_SUMMARY = _first_existing(
    DATA_ROOT
    / "06_models"
    / "selection"
    / "monthly_model_selection_summary.parquet",
    DATA_ROOT
    / "06_models"
    / "selection"
    / "monthly_prophet_model_selection_summary.parquet",
)
TEST_METRICS = _first_existing(
    DATA_ROOT / "06_models" / "selection" / "monthly_candidate_test_metrics.parquet",
    DATA_ROOT / "06_models" / "selection" / "monthly_prophet_test_metrics.parquet",
)

# Model input (05_model_input/)
ACTUALS = DATA_ROOT / "05_model_input" / "monthly_prophet_modeling_data.parquet"
RAW_EXOGENOUS = DATA_ROOT / "01_raw" / "exogenous_variables.csv"

# Inference outputs (07_model_output/)
INFERENCE_META = _first_existing(
    DATA_ROOT / "07_model_output" / "monthly_inference_metadata.json",
    DATA_ROOT / "07_model_output" / "monthly_prophet_inference_metadata.json",
)
FORECAST_LATEST = _first_existing(
    DATA_ROOT / "07_model_output" / "monthly_forecast_latest.parquet",
    DATA_ROOT / "07_model_output" / "monthly_prophet_forecast_latest.parquet",
)
ASSISTANT_VECTORSTORE = APP_CACHE_ROOT / "forecast_assistant_vectorstore"
ASSISTANT_UPLOADS = APP_CACHE_ROOT / "forecast_assistant_uploads"


def forecast_parquet(horizon_months: int) -> Path:
    """Return the path for a horizon-specific future forecast parquet."""
    return _first_existing(
        DATA_ROOT / "07_model_output" / f"monthly_forecast_{horizon_months}m.parquet",
        DATA_ROOT
        / "07_model_output"
        / f"monthly_prophet_forecast_{horizon_months}m.parquet",
    )
