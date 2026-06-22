from html import escape

import streamlit as st

from ui.components import render_hero, render_section_header
from ui.styles import apply_global_styles

st.set_page_config(
    page_title="Demand Forecast POC",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_global_styles()

render_hero(
    title="Overview",
    subtitle=(
        "A decision-support platform for demand planning: explore historical demand "
        "trends, review the monthly forecast that guides planning decisions, and "
        "understand the evidence behind each prediction."
    ),
    eyebrow="Demand Forecasting Platform",
)

st.markdown(
    """
    <p class="app-muted-text">
    Upload new demand, context, or knowledge documents to refresh the analysis, then start
    with the demand and exogenous data, walk through the monthly forecast, check how the
    models were validated, and ask the business assistant when you need a quick answer.
    Explore the sections below to begin.
    </p>
    """,
    unsafe_allow_html=True,
)

render_section_header(
    "App Sections",
    description="Select a section below to begin.",
)

_SECTIONS = [
    (
        "Data Upload",
        "Active",
        "Upload and validate demand, exogenous, and assistant-knowledge inputs.",
        "success",
        "pages/01_Data_Upload.py",
    ),
    (
        "Descriptive Analysis",
        "Active",
        "Review original demand and external-variable timelines before forecasting.",
        "success",
        "pages/02_Descriptive_Analysis.py",
    ),
    (
        "Monthly Forecast",
        "Active",
        "Review generated monthly forecast horizons, intervals, and forecast provenance.",
        "success",
        "pages/03_Monthly_Forecast.py",
    ),
    (
        "Evaluation Report",
        "Active",
        "Champion selection rationale and rolling-origin performance across model families.",
        "success",
        "pages/04_Evaluation_Report.py",
    ),
    (
        "Business Assistant",
        "Active",
        "Retrieval-grounded Q&A over forecasts, business context, and model results.",
        "success",
        "pages/05_business_assistant.py",
    ),
]

cols = st.columns(3)
for i, (name, status, desc, badge_status, page_file) in enumerate(_SECTIONS):
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
        st.page_link(page_file, label=f"Open {name}")
