"""Reusable Plotly visualisation helpers shared across the application layer.

This module provides chart-building functions shared between:
- The Streamlit app (app/) for interactive dashboards.
- The FastAPI service (api/) for generating chart JSON on demand.

All functions return Plotly Figure objects so they remain framework-agnostic —
the caller decides how to render or serialise them.
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

# Corporate color palette aligned with ui/theme.py
_COLOR_ACTUALS = "#0057B8"       # corporate blue
_COLOR_TEST = "#D97706"          # amber — test-period forecast
_COLOR_FUTURE = "#00A3E0"        # accent cyan — future forecast
_COLOR_TEST_CI = "rgba(217, 119, 6, 0.12)"
_COLOR_FUTURE_CI = "rgba(0, 163, 224, 0.12)"


def plot_forecast(
    actuals: pd.DataFrame,
    test_forecast: pd.DataFrame,
    future_forecast: pd.DataFrame,
    title: str = "Monthly Demand Forecast",
    champion_id: Optional[str] = None,
    test_mape: Optional[float] = None,
) -> go.Figure:
    """Render actuals, test-period forecast, and future forecast with prediction intervals.

    Args:
        actuals: DataFrame with columns ``ds`` (datetime) and ``y`` (float).
            Should cover the full historical window including the test period.
        test_forecast: DataFrame with ``ds``, ``y``, ``yhat``, ``yhat_lower``,
            ``yhat_upper`` for the held-out test period (champion candidate only).
        future_forecast: DataFrame with ``ds``, ``yhat``, ``yhat_lower``,
            ``yhat_upper`` for the forward-looking horizon (no actuals).
        title: Chart title displayed at the top.
        champion_id: Optional champion model identifier shown in the subtitle.
        test_mape: Optional test-set MAPE displayed in the subtitle.

    Returns:
        A Plotly Figure ready to be rendered via ``st.plotly_chart``.
    """
    fig = go.Figure()

    # --- Historical actuals ---
    fig.add_trace(
        go.Scatter(
            x=actuals["ds"],
            y=actuals["y"],
            mode="lines+markers",
            name="Actuals",
            line=dict(color=_COLOR_ACTUALS, width=2),
            marker=dict(size=5, color=_COLOR_ACTUALS),
        )
    )

    # --- Test period ---
    if not test_forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=list(test_forecast["ds"]) + list(test_forecast["ds"])[::-1],
                y=list(test_forecast["yhat_upper"]) + list(test_forecast["yhat_lower"])[::-1],
                fill="toself",
                fillcolor=_COLOR_TEST_CI,
                line=dict(color="rgba(255,255,255,0)"),
                name="Test CI (80%)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=test_forecast["ds"],
                y=test_forecast["yhat"],
                mode="lines+markers",
                name="Forecast (test)",
                line=dict(color=_COLOR_TEST, width=2, dash="dash"),
                marker=dict(size=7, symbol="diamond", color=_COLOR_TEST),
            )
        )
        _test_start = test_forecast["ds"].min().strftime("%Y-%m-%d")
        fig.add_shape(
            type="line",
            x0=_test_start,
            x1=_test_start,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#9CA3AF", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=_test_start,
            y=0.98,
            yref="paper",
            text="Test start",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color="#9CA3AF"),
        )

    # --- Future forecast ---
    if not future_forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=list(future_forecast["ds"]) + list(future_forecast["ds"])[::-1],
                y=list(future_forecast["yhat_upper"]) + list(future_forecast["yhat_lower"])[::-1],
                fill="toself",
                fillcolor=_COLOR_FUTURE_CI,
                line=dict(color="rgba(255,255,255,0)"),
                name="Forecast CI (80%)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_forecast["ds"],
                y=future_forecast["yhat"],
                mode="lines+markers",
                name="Forecast (future)",
                line=dict(color=_COLOR_FUTURE, width=2),
                marker=dict(size=6, color=_COLOR_FUTURE),
            )
        )
        _future_start = future_forecast["ds"].min().strftime("%Y-%m-%d")
        fig.add_shape(
            type="line",
            x0=_future_start,
            x1=_future_start,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color=_COLOR_FUTURE, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=_future_start,
            y=0.98,
            yref="paper",
            text="Forecast start",
            showarrow=False,
            xanchor="right",
            font=dict(size=11, color=_COLOR_FUTURE),
        )

    # --- Layout ---
    subtitle_parts: list[str] = []
    if champion_id:
        subtitle_parts.append(f"Champion: {champion_id}")
    if test_mape is not None:
        subtitle_parts.append(f"Test MAPE: {test_mape:.1%}")
    subtitle = "  |  ".join(subtitle_parts)

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
            font=dict(size=16, color="#001F5B"),
        ),
        xaxis_title="Month",
        yaxis_title="Demand (units)",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E5E7EB",
            borderwidth=1,
        ),
        template="plotly_white",
        height=500,
        margin=dict(t=80, b=40, l=60, r=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )

    return fig
