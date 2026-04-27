"""Reusable Plotly visualisation helpers shared across the application layer.

This module provides chart-building functions that are shared between:
- The Streamlit app (app/) for interactive dashboards.
- The FastAPI service (api/) for generating chart JSON on demand.

All functions in this module return Plotly Figure objects so they remain
framework-agnostic — the caller decides how to render or serialise them.

Intended implementation (Stage 3+):
- plot_forecast: renders actual vs. forecast with prediction intervals.
- plot_backtest_summary: bar chart of MAPE/RMSE across models and horizons.
- plot_hierarchy_comparison: multi-panel chart comparing monthly, weekly,
  and daily forecasts on aligned axes.
"""


def plot_forecast(artifact) -> None:
    """Render a line chart with actuals, point forecast, and prediction interval.

    Args:
        artifact: A ForecastArtifact instance from shared.schemas.

    Returns:
        plotly.graph_objects.Figure

    Raises:
        NotImplementedError: Until Stage 3 implementation.
    """
    raise NotImplementedError("plot_forecast will be implemented in Stage 3.")
