from html import escape

import streamlit as st

from ui.components import render_hero, render_section_header
from ui.page_blocks.upload_blocks import render_sidebar_upload_panel, render_upload_panel
from ui.styles import apply_global_styles

st.set_page_config(
    page_title="Demand Forecast POC",
    page_icon="📈",
    layout="wide",
)

apply_global_styles()
render_sidebar_upload_panel(key_prefix="home_sidebar")

render_hero(
    title="Demand Forecast PoC",
    subtitle=(
        "A working demand-planning cockpit for inspecting historical demand, "
        "reviewing the monthly forecast, checking model evidence, and preparing "
        "future RAG inputs."
    ),
    eyebrow="Hierarchical Demand Forecasting",
)

st.markdown(
    """
    <p class="app-muted-text">
    This application keeps the monthly planning layer as the primary business view while
    making the workflow more practical: start by reviewing historical demand and external
    variables, then move into forecast outputs, evaluation evidence, and assistant-supported
    interpretation.
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
        "Descriptive Analysis",
        "Active",
        "Review original demand and external-variable timelines before forecasting.",
        "success",
    ),
    (
        "Monthly Forecast",
        "Active",
        "Review generated monthly forecast horizons, intervals, and forecast provenance.",
        "success",
    ),
    (
        "Evaluation Report",
        "Active",
        "Champion selection rationale and held-out test performance across model families.",
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

render_upload_panel(key_prefix="home_full", compact=False)
