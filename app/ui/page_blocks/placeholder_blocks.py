"""Reusable UI blocks for scaffold/placeholder pages.

Used by: 03_weekly_anchor.py, 04_temporal_coherence.py, and any future
pages that need a structured "not yet implemented" section.
"""

from html import escape

import streamlit as st


def render_future_capability_block(
    title: str,
    status: str,
    description: str,
    expected_inputs: list[str] | None = None,
    planned_outputs: list[str] | None = None,
) -> None:
    """Render a structured card describing a planned but not yet implemented capability.

    Args:
        title: Name of the capability or feature.
        status: Short readiness label (e.g. "Planned", "In progress", "Scaffolded").
        description: One or two sentences describing what this capability will do.
        expected_inputs: Optional list of inputs this capability will consume.
        planned_outputs: Optional list of outputs this capability will produce.
    """
    badge_colors = {
        "planned": ("#EAF2FF", "#003B7A"),
        "scaffolded": ("#FFF7D6", "#92400E"),
        "in progress": ("#E8F7EF", "#065F46"),
        "blocked": ("#FDECEC", "#991B1B"),
    }
    bg, color = badge_colors.get(status.lower(), ("#F3F4F6", "#374151"))

    inputs_html = ""
    if expected_inputs:
        items = "".join(f"<li>{escape(i)}</li>" for i in expected_inputs)
        inputs_html = f"""
        <div class="app-capability-label">Expected inputs</div>
        <ul class="app-capability-block-list">{items}</ul>
        """

    outputs_html = ""
    if planned_outputs:
        items = "".join(f"<li>{escape(o)}</li>" for o in planned_outputs)
        outputs_html = f"""
        <div class="app-capability-label" style="margin-top:0.6rem;">Planned outputs</div>
        <ul class="app-capability-block-list">{items}</ul>
        """

    st.markdown(
        f"""
        <div class="app-capability-block">
            <div class="app-capability-block-title">
                {escape(title)}
                <span style="
                    font-size: 0.68rem;
                    font-weight: 700;
                    letter-spacing: 0.06em;
                    text-transform: uppercase;
                    background: {bg};
                    color: {color};
                    border-radius: 999px;
                    padding: 0.18rem 0.55rem;
                ">{escape(status)}</span>
            </div>
            <div class="app-capability-block-description">{escape(description)}</div>
            {inputs_html}
            {outputs_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
