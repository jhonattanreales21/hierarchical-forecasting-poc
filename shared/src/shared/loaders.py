"""Loaders for forecast artifacts and model outputs.

This module is responsible for reading ForecastArtifact objects from the
file system — specifically from data/07_model_output/ (JSON artifacts) and
data/08_reporting/ (evaluation summaries) — and deserialising them into the
shared Pydantic schemas defined in shared.schemas.

It will be consumed by:
- The Streamlit app (app/) to populate dashboards and charts.
- The FastAPI service (api/) to serve forecast data on demand.

Intended implementation (Stage 3+):
- load_forecast_output: reads a versioned JSON file by model name and
  granularity, parses it into a ForecastArtifact, and returns it.
- list_available_runs: scans the output directory and returns metadata
  for all available artifacts.
"""

from pathlib import Path


def load_forecast_output(
    model_name: str,
    granularity: str,
    output_dir: Path | str = "pipelines/data/07_model_output",
) -> None:
    """Load a ForecastArtifact from disk for a given model and granularity.

    Args:
        model_name: Name of the model (e.g. "catboost_weekly").
        granularity: Temporal granularity string (e.g. "weekly").
        output_dir: Root directory where artifacts are stored.

    Returns:
        ForecastArtifact parsed from JSON.

    Raises:
        NotImplementedError: Until Stage 3 implementation.
    """
    raise NotImplementedError("load_forecast_output will be implemented in Stage 3.")
