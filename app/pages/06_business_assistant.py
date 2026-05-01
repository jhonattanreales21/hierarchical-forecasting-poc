import streamlit as st

from ui.components import render_page_header
from ui.page_blocks.assistant_blocks import (
    render_assistant_status,
    render_business_assistant_intro,
    render_disabled_chat_mockup,
    render_future_integration_approach,
    render_planned_capabilities,
)
from ui.styles import apply_global_styles

apply_global_styles()

render_page_header(
    title="Business Assistant",
    subtitle="Planned natural-language Q&A layer over forecasts and business context.",
    eyebrow="Future Capability",
)

st.info(
    "This page is a controlled visual scaffold. "
    "No LLM, RAG, or retrieval logic is active in this version."
)

left_col, right_col = st.columns([1.7, 1.0], gap="large")

with left_col:
    render_business_assistant_intro()
    render_disabled_chat_mockup()

with right_col:
    render_assistant_status()

render_future_integration_approach()
render_planned_capabilities()
