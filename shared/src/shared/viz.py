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
_COLOR_ACTUALS = "#0057B8"  # corporate blue
_COLOR_TEST = "#D97706"  # amber — test-period forecast
_COLOR_FUTURE = "#00A3E0"  # accent cyan — future forecast
_COLOR_TEST_CI = "rgba(217, 119, 6, 0.12)"
_COLOR_FUTURE_CI = "rgba(0, 163, 224, 0.12)"


def plot_forecast(
    actuals: pd.DataFrame,
    test_forecast: pd.DataFrame,
    future_forecast: pd.DataFrame,
    title: str = "Monthly Demand Forecast",
    champion_id: Optional[str] = None,
    test_mape: Optional[float] = None,
    show_future_intervals: bool = True,
    show_test_intervals: bool = True,
) -> go.Figure:
    """Render actuals, test-period forecast, and future forecast with prediction intervals.

    Interval bands are model-family aware: pass ``show_future_intervals=False``
    (and/or ``show_test_intervals=False``) for champions that do not produce
    prediction intervals, so the chart shows point forecasts only without
    implying spurious uncertainty bounds.

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
        show_future_intervals: When True, draw the future prediction-interval band.
        show_test_intervals: When True, draw the test-period interval band.

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
        _has_test_bounds = {"yhat_lower", "yhat_upper"}.issubset(test_forecast.columns)
        if show_test_intervals and _has_test_bounds:
            fig.add_trace(
                go.Scatter(
                    x=list(test_forecast["ds"]) + list(test_forecast["ds"])[::-1],
                    y=list(test_forecast["yhat_upper"])
                    + list(test_forecast["yhat_lower"])[::-1],
                    fill="toself",
                    fillcolor=_COLOR_TEST_CI,
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Test interval",
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
        _has_future_bounds = {"yhat_lower", "yhat_upper"}.issubset(
            future_forecast.columns
        )
        if show_future_intervals and _has_future_bounds:
            fig.add_trace(
                go.Scatter(
                    x=list(future_forecast["ds"]) + list(future_forecast["ds"])[::-1],
                    y=list(future_forecast["yhat_upper"])
                    + list(future_forecast["yhat_lower"])[::-1],
                    fill="toself",
                    fillcolor=_COLOR_FUTURE_CI,
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Forecast interval",
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


def add_event_lines(
    fig: go.Figure,
    events: pd.DataFrame,
    event_columns: dict[str, str],
    date_col: str = "ds",
    colors: Optional[list[str]] = None,
) -> go.Figure:
    """Overlay vertical reference lines on periods flagged by binary event columns.

    For each entry in ``event_columns`` (mapping a flag column name to a human
    label), a dashed vertical line is drawn at every period where the flag is
    truthy (numeric value > 0). One invisible legend proxy per event type is added
    so the line colors are explained in the legend without labelling each line.

    Args:
        fig: Target Plotly figure, e.g. one returned by ``plot_forecast``.
        events: DataFrame with a datetime column and one or more binary flag columns.
        event_columns: Ordered mapping of flag column name -> legend label.
        date_col: Name of the datetime column in ``events``.
        colors: Optional list of line colors, one per event type (cycled if shorter).

    Returns:
        The same figure with vertical event lines and legend proxies added.
    """
    if events is None or events.empty or date_col not in events.columns:
        return fig

    palette = colors or ["#F59E0B", "#7C3AED", "#0EA5E9", "#EF4444"]
    dates = pd.to_datetime(events[date_col], errors="coerce")

    for idx, (column, label) in enumerate(event_columns.items()):
        if column not in events.columns:
            continue
        color = palette[idx % len(palette)]
        flags = pd.to_numeric(events[column], errors="coerce").fillna(0) > 0
        flagged_dates = dates[flags].dropna().drop_duplicates()
        if flagged_dates.empty:
            continue
        for ts in flagged_dates:
            ts_str = ts.strftime("%Y-%m-%d")
            fig.add_shape(
                type="line",
                x0=ts_str,
                x1=ts_str,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.65,
            )
        # Legend proxy: invisible trace so the color is explained in the legend.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, width=2, dash="dot"),
                name=label,
            )
        )
    return fig


def plot_feature_importance_bar(
    importance: pd.DataFrame,
    title: str = "Champion Feature Importance",
    top_n: int = 15,
    importance_col: str = "importance",
    feature_col: str = "feature",
    subtitle: Optional[str] = None,
    x_axis_title: str = "Importance",
) -> go.Figure:
    """Render a horizontal bar chart of feature/driver importance for a champion model.

    Family-agnostic: the importance values may be SHAP mean(|value|) (CatBoost), centered
    component contributions (Prophet), or absolute coefficients (SARIMAX). The caller sets
    ``x_axis_title``/``subtitle`` so the axis labelling stays honest about the statistic.

    Args:
        importance: DataFrame with at least ``feature_col`` and ``importance_col``.
        title: Chart title.
        top_n: Keep only the ``top_n`` most important features.
        importance_col: Name of the importance value column.
        feature_col: Name of the feature label column.
        subtitle: Optional subtitle shown under the title (e.g. champion id + method).
        x_axis_title: Label for the value axis (statistic-specific).

    Returns:
        A Plotly Figure ready to be rendered via ``st.plotly_chart``.
    """
    fig = go.Figure()

    if importance is not None and not importance.empty:
        ranked = (
            importance.sort_values(importance_col, ascending=False)
            .head(top_n)
            # Plotly draws the first row at the bottom; reverse so the largest is on top.
            .iloc[::-1]
        )
        fig.add_trace(
            go.Bar(
                x=ranked[importance_col],
                y=ranked[feature_col].astype(str),
                orientation="h",
                marker=dict(color=_COLOR_ACTUALS),
                hovertemplate="%{y}: %{x:.4g}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
            font=dict(size=16, color="#001F5B"),
        ),
        xaxis_title=x_axis_title,
        yaxis_title="",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(showgrid=False, automargin=True),
        template="plotly_white",
        height=max(320, 28 * min(len(importance) if importance is not None else 0, top_n) + 120),
        margin=dict(t=80, b=40, l=20, r=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
    )

    return fig
