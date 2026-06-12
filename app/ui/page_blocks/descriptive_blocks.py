"""Page-specific UI blocks for historical demand descriptive analysis."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components import (
    render_empty_state,
    render_kpi_card,
    render_page_header,
    render_section_header,
)
from utils.descriptive import (
    Granularity,
    exogenous_value_columns,
    filter_demand,
    filter_exogenous,
    summarize_demand,
)
from utils.formatting import format_metric

_GRANULARITY_LABELS: dict[Granularity, str] = {
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}


def render_descriptive_page_header() -> None:
    """Render the descriptive analysis page header."""
    render_page_header(
        title="Descriptive Analysis",
        subtitle=(
            "Inspect original demand observations and external-variable timelines "
            "before reviewing model forecasts."
        ),
        eyebrow="Historical Data Review",
    )


def render_descriptive_controls(
    demand_by_granularity: dict[Granularity, pd.DataFrame],
) -> tuple[Granularity, str | None, pd.Timestamp, pd.Timestamp]:
    """Render SKU, granularity, and date controls."""
    available = [
        granularity
        for granularity in ("daily", "weekly", "monthly")
        if not demand_by_granularity[granularity].empty
    ]
    if not available:
        return "monthly", None, pd.Timestamp.today(), pd.Timestamp.today()

    with st.container(key="descriptive-controls", border=True):
        st.markdown(
            """
            <div class="monthly-toolbar-header">
                <div class="monthly-toolbar-label">Data Controls</div>
                <div class="monthly-toolbar-title">Demand review filters</div>
                <p class="monthly-toolbar-copy">
                    Choose the SKU, temporal detail, and date window for historical demand.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns([1.0, 1.0, 1.6])
        with cols[0]:
            granularity = st.selectbox(
                "Level of detail",
                options=available,
                format_func=lambda value: _GRANULARITY_LABELS[value],
                index=available.index("monthly") if "monthly" in available else 0,
                key="descriptive_granularity",
            )
        base = demand_by_granularity[granularity]
        sku_options = sorted(base["sku"].dropna().astype(str).unique())
        with cols[1]:
            sku = st.selectbox(
                "SKU",
                options=sku_options,
                index=0,
                key="descriptive_sku",
            )
        min_date = pd.Timestamp(base["date"].min()).to_pydatetime()
        max_date = pd.Timestamp(base["date"].max()).to_pydatetime()
        with cols[2]:
            if min_date == max_date:
                st.caption(f"Date: {min_date:%Y-%m-%d}")
                start_date = end_date = pd.Timestamp(min_date)
            else:
                selected_range = st.slider(
                    "Date range",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date),
                    format="YYYY-MM-DD",
                    key="descriptive_date_range",
                )
                start_date = pd.Timestamp(selected_range[0])
                end_date = pd.Timestamp(selected_range[1])

    return granularity, sku, start_date, end_date


def render_demand_summary(df: pd.DataFrame, granularity: Granularity) -> None:
    """Render demand summary KPI cards."""
    summary = summarize_demand(df)
    label = _GRANULARITY_LABELS[granularity].lower()
    cols = st.columns(4)
    with cols[0]:
        render_kpi_card(
            label="Total demand",
            value=format_metric(summary.total_demand, decimals=0, suffix=" units"),
        )
    with cols[1]:
        render_kpi_card(
            label=f"Average {label}",
            value=format_metric(summary.average_demand, decimals=1, suffix=" units"),
        )
    with cols[2]:
        render_kpi_card(
            label=f"Peak {label}",
            value=format_metric(summary.peak_demand, decimals=1, suffix=" units"),
        )
    with cols[3]:
        render_kpi_card(
            label="Periods",
            value=str(summary.period_count),
            help_text="Rows after current filters",
        )


def render_demand_timeline(df: pd.DataFrame, granularity: Granularity, sku: str) -> None:
    """Render historical demand timeline."""
    render_section_header(
        "Demand Timeline",
        description="Original demand observations at the selected temporal granularity.",
    )
    if df.empty:
        render_empty_state(
            title="No Demand Data In This Window",
            message="Adjust the SKU, level of detail, or date range.",
            status="warning",
        )
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["demand"],
            mode="lines+markers",
            name=f"{_GRANULARITY_LABELS[granularity]} demand",
            line={"color": "#0057B8", "width": 2.5},
            marker={"size": 6},
            hovertemplate="%{x|%Y-%m-%d}<br>Demand: %{y:,.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{escape(sku)} demand ({_GRANULARITY_LABELS[granularity].lower()})",
        height=430,
        margin={"l": 20, "r": 20, "t": 60, "b": 35},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis_title="Date",
        yaxis_title="Demand units",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def render_exogenous_timeline(
    exogenous: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Render monthly exogenous variables aligned to the selected window."""
    render_section_header(
        "External Variables Timeline",
        description="Monthly exogenous variables available to the forecasting workflow.",
    )
    filtered = filter_exogenous(exogenous, start_date, end_date)
    value_cols = exogenous_value_columns(filtered)
    if filtered.empty or not value_cols:
        st.info("No exogenous-variable values are available for the selected window.")
        return

    selected_cols = st.multiselect(
        "Variables to display",
        options=value_cols,
        default=value_cols[: min(4, len(value_cols))],
        key="descriptive_exogenous_vars",
    )
    if not selected_cols:
        st.info("Select at least one external variable to display.")
        return

    fig = go.Figure()
    colors = ["#0057B8", "#00A3E0", "#10B981", "#D97706", "#7C3AED", "#EF4444"]
    for idx, col in enumerate(selected_cols):
        fig.add_trace(
            go.Scatter(
                x=filtered["date"],
                y=filtered[col],
                mode="lines+markers",
                name=col.replace("_", " ").title(),
                line={"color": colors[idx % len(colors)], "width": 2},
                hovertemplate=f"{col}: %{{y:,.3g}}<extra></extra>",
            )
        )
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 25, "b": 35},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis_title="Month",
        yaxis_title="Value",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.25},
    )
    st.plotly_chart(fig, width="stretch")


def render_demand_table(df: pd.DataFrame, granularity: Granularity) -> None:
    """Render filtered demand data table and CSV download."""
    render_section_header(
        "Filtered Demand Data",
        description="Rows behind the selected historical review.",
    )
    if df.empty:
        st.info("No rows to display.")
        return

    display = df.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display = display.rename(
        columns={
            "date": "Date",
            "sku": "SKU",
            "demand": "Demand",
            "granularity": "Granularity",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        label="Download filtered demand CSV",
        data=df.to_csv(index=False),
        file_name=f"descriptive_demand_{granularity}.csv",
        mime="text/csv",
    )


def render_descriptive_analysis(
    demand_by_granularity: dict[Granularity, pd.DataFrame],
    exogenous: pd.DataFrame,
) -> None:
    """Render the complete descriptive-analysis workflow."""
    if all(frame.empty for frame in demand_by_granularity.values()):
        render_empty_state(
            title="No Historical Demand Artifacts Found",
            message=(
                "Run the data-ingestion pipeline to generate demand_daily, "
                "demand_weekly, and demand_monthly artifacts."
            ),
            status="warning",
        )
        return

    granularity, sku, start_date, end_date = render_descriptive_controls(
        demand_by_granularity
    )
    if sku is None:
        return

    filtered = filter_demand(
        demand_by_granularity[granularity],
        sku=sku,
        start_date=start_date,
        end_date=end_date,
    )
    render_demand_summary(filtered, granularity)
    render_demand_timeline(filtered, granularity, sku)
    render_exogenous_timeline(exogenous, start_date, end_date)
    render_demand_table(filtered, granularity)
