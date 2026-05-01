"""Page-specific UI blocks for the Monthly Forecast page (02_monthly_forecast.py).

All functions receive already-loaded data and have no knowledge of artifact paths.
The page script (02_monthly_forecast.py) handles all data loading and orchestration.
"""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components import (
    render_empty_state,
    render_hero,
    render_info_banner,
    render_kpi_card,
    render_section_header,
    render_warning_banner,
)
from utils.formatting import (
    format_date,
    format_metric,
    format_optional,
    format_percentage,
)
from utils.paths import forecast_parquet

_HORIZON_LABELS: dict[int, str] = {3: "3 months", 6: "6 months", 12: "12 months"}
_ALL_HORIZONS: list[int] = [3, 6, 12]


def _detect_available_horizons() -> list[int]:
    """Return the list of horizons (in months) for which forecast parquets exist."""
    return [h for h in _ALL_HORIZONS if forecast_parquet(h).exists()]


def render_monthly_page_header() -> None:
    """Render the hero banner for the Monthly Forecast page."""
    render_hero(
        title="Monthly Forecast — Primary Decision Layer",
        subtitle=(
            "Forward-looking monthly demand forecast generated from the current Prophet champion "
            "model. This is the primary forecast consumption view and the main executive-facing "
            "layer of the MVP."
        ),
        eyebrow="Hierarchical Demand Forecasting · Monthly Layer",
    )


def render_monthly_section_gap() -> None:
    """Render a compact spacer between monthly page sections."""
    st.markdown('<div class="monthly-section-gap"></div>', unsafe_allow_html=True)


def render_monthly_kpi_summary(
    meta: dict,
    inference_meta: dict,
    horizon_months: int,
) -> None:
    """Render a row of KPI cards summarising the champion model and current forecast.

    Args:
        meta: Champion model metadata dict.
        inference_meta: Inference run metadata dict.
        horizon_months: Currently selected forecast horizon in months.
    """
    render_section_header("Champion Model Summary")

    test_metrics = meta.get("test_metrics", {})
    mape = test_metrics.get("mape")
    rmse = test_metrics.get("rmse")
    precision = test_metrics.get("forecast_precision")
    business_flag = meta.get("business_success_flag", False)
    champion_id = meta.get("champion_id", "N/A")
    created_at = inference_meta.get("forecast_created_at", "")
    last_run = format_date(created_at) if created_at else "Not available"

    mape_status = "success" if (mape is not None and mape < 0.15) else "warning"
    precision_status = "success" if business_flag else "warning"

    cols = st.columns(5)
    with cols[0]:
        render_kpi_card(
            label="Champion Model",
            value=format_optional(champion_id),
            help_text="Current monthly Prophet champion",
        )
    with cols[1]:
        render_kpi_card(
            label="Test MAPE",
            value=format_percentage(mape) if mape is not None else "N/A",
            help_text="Mean Absolute % Error on test set",
            status=mape_status,
        )
    with cols[2]:
        render_kpi_card(
            label="Test RMSE",
            value=(
                format_metric(rmse, decimals=1, suffix=" units")
                if rmse is not None
                else "N/A"
            ),
            help_text="Root Mean Squared Error on test set",
        )
    with cols[3]:
        render_kpi_card(
            label="Forecast Precision",
            value=format_percentage(precision) if precision is not None else "N/A",
            help_text="1 − MAPE; business target ≥ 85%",
            status=precision_status,
        )
    with cols[4]:
        render_kpi_card(
            label="Selected Horizon",
            value=f"{horizon_months} months",
            help_text=f"Last forecast run: {last_run}",
        )


def render_horizon_selector() -> int:
    """Render the forecast horizon selector and return the selected horizon in months.

    Detects available horizons from existing forecast parquet files. If no horizons
    are found, falls back to a default of 6 months and shows a warning.

    Returns:
        Selected forecast horizon in months.
    """
    available = _detect_available_horizons()

    if not available:
        with st.container(key="monthly-horizon-toolbar", border=True):
            st.markdown(
                """
                <div class="monthly-toolbar-header">
                    <div class="monthly-toolbar-label">Forecast Controls</div>
                    <div class="monthly-toolbar-title">Forecast horizon</div>
                    <p class="monthly-toolbar-copy">
                        No generated forecast horizons are currently available, so the page will
                        fall back to the default 6-month view until new artifacts are produced.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.warning(
                "No forecast horizons found. "
                "Run `uv run kedro run --pipeline forecast_inference` to generate forecasts."
            )
        return 6

    options = [_HORIZON_LABELS[h] for h in available]
    default_idx = len(options) - 1  # longest available horizon as default
    missing = [h for h in _ALL_HORIZONS if h not in available]

    with st.container(key="monthly-horizon-toolbar", border=True):
        st.markdown(
            """
            <div class="monthly-toolbar-header">
                <div class="monthly-toolbar-label">Forecast Controls</div>
                <div class="monthly-toolbar-title">Forecast horizon</div>
                <p class="monthly-toolbar-copy">
                    Choose the executive view to compare near-term and longer-range demand outlooks.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        widget_help = (
            "Select the forward-looking forecast window to display on this page."
        )
        if hasattr(st, "segmented_control"):
            selected_label = st.segmented_control(
                "Forecast horizon",
                options=options,
                default=options[default_idx],
                selection_mode="single",
                key="monthly_forecast_horizon",
                help=widget_help,
                label_visibility="collapsed",
                width="stretch",
            )
        else:
            selected_label = st.radio(
                "Forecast horizon",
                options=options,
                index=default_idx,
                horizontal=True,
                key="monthly_forecast_horizon",
                help=widget_help,
                label_visibility="collapsed",
            )

        if missing:
            missing_labels = ", ".join(_HORIZON_LABELS[h] for h in missing)
            st.markdown(
                f"""
                <p class="monthly-toolbar-note">
                    <strong>Unavailable right now:</strong> {escape(missing_labels)}.
                    The selector only shows horizons that have generated artifacts.
                </p>
                """,
                unsafe_allow_html=True,
            )

    label_to_months = {v: k for k, v in _HORIZON_LABELS.items()}
    return label_to_months[selected_label or options[default_idx]]


def render_forecast_chart_panel(
    fig: go.Figure,
    meta: dict,
    horizon_months: int,
) -> None:
    """Render the main forecast chart inside a styled section panel.

    Args:
        fig: Plotly figure returned by plot_forecast().
        meta: Champion metadata dict used for the caption.
        horizon_months: Selected horizon in months.
    """
    train_start = format_date(meta.get("train_window", {}).get("start_date", ""))
    train_end = format_date(meta.get("train_window", {}).get("end_date", ""))
    test_start = format_date(meta.get("test_window", {}).get("start_date", ""))
    test_end = format_date(meta.get("test_window", {}).get("end_date", ""))

    caption_parts = []
    if train_start and train_end:
        caption_parts.append(f"Training: {train_start} → {train_end}")
    if test_start and test_end:
        caption_parts.append(f"Test: {test_start} → {test_end}")
    caption_parts.append(f"Horizon: {horizon_months} months forward")

    with st.container(key="monthly-chart-panel", border=True):
        st.markdown(
            f"""
            <div class="monthly-chart-panel-header">
                <div class="monthly-chart-panel-label">Primary View</div>
                <div class="monthly-chart-panel-title">Demand Forecast Chart</div>
                <p class="monthly-chart-panel-copy">
                    Monthly actuals, test-period backtesting, and the selected
                    {horizon_months}-month forward forecast with 80% prediction intervals.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"""
            <div class="monthly-chart-panel-meta">
                {escape(" | ".join(caption_parts))}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_executive_forecast_summary(
    meta: dict,
    inference_meta: dict,
    horizon_months: int,
    future_fc: pd.DataFrame,
) -> None:
    """Render a business-facing narrative interpretation of the current forecast.

    Args:
        meta: Champion model metadata dict.
        inference_meta: Inference run metadata dict.
        horizon_months: Currently selected forecast horizon in months.
        future_fc: Future forecast DataFrame (may be empty).
    """
    render_section_header(
        "Executive Interpretation",
        description="Business-facing summary of the current forecast output.",
    )

    test_metrics = meta.get("test_metrics", {})
    precision = test_metrics.get("forecast_precision")
    business_flag = meta.get("business_success_flag", False)
    champion_id = meta.get("champion_id", "the champion model")
    precision_threshold = meta.get("business_success_precision_threshold", 0.85)

    precision_str = format_percentage(precision) if precision is not None else "N/A"
    target_str = (
        format_percentage(precision_threshold) if precision_threshold else "85.0%"
    )

    if business_flag:
        status_line = (
            f"The model has met the business precision target of {target_str}, "
            f"achieving {precision_str} on the held-out test set."
        )
    else:
        status_line = (
            f"Current forecast precision ({precision_str}) is below the business target "
            f"of {target_str}. This output should be interpreted as a functional MVP baseline "
            "rather than a final production-ready model."
        )

    if not future_fc.empty and "yhat" in future_fc.columns:
        avg_f = future_fc["yhat"].mean()
        min_f = future_fc["yhat"].min()
        max_f = future_fc["yhat"].max()
        range_line = (
            f"Over the selected **{horizon_months}-month horizon**, projected demand ranges "
            f"from **{min_f:,.0f}** to **{max_f:,.0f}** units "
            f"(average: **{avg_f:,.0f}** units/month)."
        )
    else:
        range_line = (
            f"No future forecast data is currently available for the "
            f"{horizon_months}-month horizon."
        )

    st.markdown(f"""
The active monthly Prophet champion (*{escape(champion_id)}*) provides the current forecast
for the {horizon_months}-month horizon. {status_line}

{range_line}

> **Note:** This page reflects the Prophet MVP baseline. Forecast precision targets and model
> candidates will continue to evolve as the project progresses toward production readiness.
        """)


def render_future_forecast_table(
    future_fc: pd.DataFrame,
    horizon_months: int,
) -> None:
    """Render the future forecast values in a clean, stakeholder-friendly table.

    Args:
        future_fc: Future forecast DataFrame with ds, yhat, yhat_lower, yhat_upper columns.
        horizon_months: Selected horizon in months (used for headings and download filename).
    """
    render_section_header(
        f"Forecast Table — {horizon_months}-Month Horizon",
        description="Projected monthly demand values with 80% prediction intervals.",
    )

    if future_fc.empty:
        render_empty_state(
            title="No Forecast Data Available",
            message=(
                f"The {horizon_months}-month forecast artifact was not found. "
                "Run `uv run kedro run --pipeline forecast_inference` to generate it."
            ),
            status="warning",
        )
        return

    available_cols = set(future_fc.columns)
    ordered_cols = [
        c
        for c in ["ds", "horizon_month", "yhat", "yhat_lower", "yhat_upper"]
        if c in available_cols
    ]
    display = future_fc[ordered_cols].copy()

    if "ds" in display.columns:
        display["ds"] = display["ds"].dt.strftime("%b %Y")

    rename_map = {
        "ds": "Month",
        "horizon_month": "Horizon",
        "yhat": "Forecast (units)",
        "yhat_lower": "Lower Bound (80%)",
        "yhat_upper": "Upper Bound (80%)",
    }
    display = display.rename(
        columns={k: v for k, v in rename_map.items() if k in display.columns}
    )

    for col in ["Forecast (units)", "Lower Bound (80%)", "Upper Bound (80%)"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda v: f"{v:,.0f}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    csv_data = future_fc.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"monthly_prophet_forecast_{horizon_months}m.csv",
        mime="text/csv",
    )


def render_monthly_data_status() -> None:
    """Render a data availability note showing which forecast horizons are available."""
    available = _detect_available_horizons()
    missing = [h for h in _ALL_HORIZONS if h not in available]

    if missing:
        missing_str = ", ".join(f"{h}m" for h in missing)
        render_warning_banner(
            title="Some Forecast Horizons Unavailable",
            message=(
                f"The following horizons are not yet generated: {missing_str}. "
                "Run `uv run kedro run --pipeline forecast_inference` to generate all horizons."
            ),
        )
    else:
        render_info_banner(
            title="All Forecast Horizons Available",
            message=(
                "3-month, 6-month, and 12-month forecasts have all been generated "
                "and are ready for display."
            ),
        )
