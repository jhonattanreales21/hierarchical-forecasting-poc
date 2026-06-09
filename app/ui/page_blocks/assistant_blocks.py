"""Page-specific UI blocks for the Business Assistant page (06_business_assistant.py).

All CSS used here is defined in styles.py (applied via apply_global_styles()).
No LLM, RAG, vector store, or API calls belong in any of these functions.
"""

from html import escape

import streamlit as st

from ui.components import render_section_header


def render_business_assistant_intro() -> None:
    """Render the intro card describing the business assistant concept."""
    st.markdown(
        """
        <div class="app-card">
            <div class="app-section-title"
                 style="border:none; padding:0; margin-bottom:0.4rem;">
                Chat-based layer for business questions
            </div>
            <p class="app-muted-text">
                The idea is simple: executives ask a question about the forecast and the
                assistant answers using business documents, forecast outputs, historical
                results, and model explanations — all through natural language.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disabled_chat_mockup() -> None:
    """Render a visual-only chat preview with a disabled input field."""
    render_section_header(
        "Chat Preview",
        description="Visual scaffold only — prompting is disabled in this version.",
    )
    st.markdown(
        """
        <div class="app-chat-shell">
            <div class="app-chat-bubble user">
                <div class="app-chat-label">Executive prompt</div>
                Why is the latest monthly forecast softer than the previous view?
            </div>
            <div class="app-chat-bubble assistant">
                <div class="app-chat-label">Business assistant</div>
                The answer would explain the forecast change in plain business language,
                point to the most relevant evidence, and summarise the likely planning
                impact. <em>(Static mockup — no LLM is active.)</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    input_col, button_col = st.columns([5.4, 1.2], gap="small")
    with input_col:
        st.text_input(
            "Assistant prompt",
            value="Ask about forecast changes, risks, or drivers...",
            disabled=True,
            label_visibility="collapsed",
        )
    with button_col:
        st.button("Send", disabled=True, width="stretch")
    st.caption(
        "Prompting is disabled on purpose. This page is a visual scaffold for a future RAG assistant."
    )


def render_assistant_status() -> None:
    """Render the readiness map showing integration layer status."""
    render_section_header(
        "Readiness Map",
        description="Integration layers required before the assistant can respond to real queries.",
    )
    items = [
        (
            "Business documents",
            "Planned RAG source",
            "Decks, planning notes, and assumptions would provide business context.",
            "info",
        ),
        (
            "Forecast outputs",
            "Available in platform",
            "Monthly and weekly outputs can become the evidence layer behind answers.",
            "success",
        ),
        (
            "Historical results",
            "Needs curation",
            "Prior runs and realized demand would support comparisons over time.",
            "warning",
        ),
        (
            "Model explanations",
            "Designed, not wired",
            "Model logic should be translated into concise business-facing explanations.",
            "info",
        ),
    ]
    for title, badge, copy, badge_status in items:
        st.markdown(
            f"""
            <div class="app-readiness-card">
                <div class="app-readiness-row">
                    <div class="app-readiness-title">{escape(title)}</div>
                    <span class="app-status-badge {badge_status}">{escape(badge)}</span>
                </div>
                <p class="app-readiness-copy">{escape(copy)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_planned_capabilities() -> None:
    """Render the list of planned assistant capabilities."""
    render_section_header(
        "Planned Capabilities",
        description="What the business assistant will be able to answer once deployed.",
    )
    st.markdown("""
- Explain forecast changes between planning cycles in plain business language
- Summarise key demand drivers and model signals for a given period
- Compare current forecast against historical averages and seasonal patterns
- Flag anomalies or unusually wide prediction intervals
- Answer questions about evaluation results and model selection rationale
        """)


def render_future_integration_approach() -> None:
    """Render a brief explanation of the planned RAG integration architecture."""
    render_section_header(
        "Integration Architecture",
        description="How the assistant will be wired once implemented.",
    )
    st.markdown("""
The assistant is designed as a **retrieval-augmented generation (RAG)** pipeline:

1. **Document ingestion** — business documents, historical forecasts, and model summaries
   are indexed and stored in a vector database.
2. **Retrieval** — relevant chunks are fetched at query time based on semantic similarity.
3. **LLM orchestration** — a language model generates grounded answers using the retrieved
   context, constrained to the available evidence.
4. **Streamlit interface** — the existing chat UI serves as the front end.

The retrieval, embedding, and LLM orchestration layers live outside this page — behind
a FastAPI endpoint or a dedicated module — keeping the Streamlit script lightweight.
        """)
