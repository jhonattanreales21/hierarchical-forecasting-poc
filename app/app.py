from html import escape

import streamlit as st

from ui.components import render_hero, render_section_header
from ui.styles import apply_global_styles

st.set_page_config(
    page_title="Demand Forecast POC",
    page_icon="📈",
    layout="wide",
)

apply_global_styles()

render_hero(
    title="Demand Forecast PoC",
    subtitle=(
        "Temporal hierarchical forecasting for a critical SKU. "
        "Explore the monthly forecast, model evaluation, project status, "
        "and future business assistant capabilities from the sidebar."
    ),
    eyebrow="Hierarchical Demand Forecasting",
)

st.markdown(
    """
    <p class="app-muted-text">
    This PoC demonstrates a complete forecasting system — from raw demand data and exogenous
    variables through a Kedro ML pipeline to evaluated forecasts served through this Streamlit
    application. The <strong>monthly layer is the primary, business-facing layer</strong>: it
    displays the current production champion selected from a time-aware competition among
    Prophet, SARIMAX, and CatBoost. Weekly and reconciliation layers are planned enhancements.
    </p>
    """,
    unsafe_allow_html=True,
)

render_section_header(
    "App Sections",
    description="Select a page from the sidebar to navigate.",
)

_SECTIONS = [
    (
        "Monthly Forecast",
        "Active",
        "Forward-looking monthly demand forecast from the current production champion.",
        "success",
    ),
    (
        "Evaluation Report",
        "Active",
        "Champion selection rationale and held-out test performance across model families.",
        "success",
    ),
    (
        "Project Overview",
        "Active",
        "Live champion status, pipeline artifact health, and PoC scope summary.",
        "success",
    ),
    (
        "Business Assistant",
        "Active",
        "Retrieval-grounded Q&A over forecasts, business context, and model results.",
        "success",
    ),
]

cols = st.columns(3)
for i, (name, status, desc, badge_status) in enumerate(_SECTIONS):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="app-nav-card">
                <div class="app-nav-card-title">{escape(name)}</div>
                <div class="app-nav-card-description">{escape(desc)}</div>
                <span class="app-status-badge {badge_status}">{escape(status)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
