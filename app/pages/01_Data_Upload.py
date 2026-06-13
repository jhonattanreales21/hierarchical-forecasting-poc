import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

from ui.components import render_page_header
from ui.navigation import render_top_navbar
from ui.page_blocks.upload_blocks import render_data_upload_page
from ui.styles import apply_global_styles

apply_global_styles()
render_top_navbar("Data Upload")
render_page_header(
    title="Data Upload",
    subtitle=(
        "Upload demand and exogenous CSV data, plus business-history documents for "
        "the assistant. Files are validated and stored; no pipeline is triggered."
    ),
    eyebrow="Inputs Intake",
)

render_data_upload_page()
