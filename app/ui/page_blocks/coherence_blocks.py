"""Page-specific UI blocks for the Temporal Coherence page (04_temporal_coherence.py).

Each function renders a self-contained section of the page. No pipeline logic or
artifact path references belong here — those stay in the page script.
"""

import streamlit as st

from ui.components import render_section_header, render_warning_banner
from ui.page_blocks.placeholder_blocks import render_future_capability_block


def render_temporal_coherence_intro() -> None:
    """Render the introductory explanation of temporal coherence with a hierarchy table."""
    render_section_header(
        "What is Temporal Coherence?",
        description=(
            "Coherence means forecasts at different time granularities are internally consistent: "
            "the sum of weekly forecasts within a month must align with the monthly total."
        ),
    )
    st.markdown(
        """
| Layer | Role | Coherence target |
|-------|------|-----------------|
| **Monthly** | Primary decision layer | Reference (top level) |
| **Weekly** | Secondary operational layer | Must sum to monthly |
| **Daily** | Optional extension (disabled) | Must sum to weekly |
        """
    )
    st.caption(
        "Default reconciliation method: mint_shrink  "
        "(configured in conf/base/parameters/reconciliation.yml)"
    )


def render_reconciliation_status() -> None:
    """Render the current reconciliation pipeline status and known constraints."""
    render_section_header(
        "Current Status",
        description="Reconciliation pipeline readiness and active configuration.",
    )
    render_warning_banner(
        title="Reconciliation not yet active",
        message=(
            "The monthly layer is the active MVP. Weekly training and reconciliation "
            "outputs will be generated once the weekly pipeline is fully validated."
        ),
    )
    render_warning_banner(
        title="Daily allocation disabled",
        message=(
            "Daily allocation is currently disabled. "
            "Re-enable by setting `daily_allocation.enabled: true` "
            "in `conf/base/parameters/forecast_inference.yml`."
        ),
    )


def render_expected_reconciliation_artifacts() -> None:
    """Render the planned reconciliation diagnostics as a capability block."""
    render_future_capability_block(
        title="Reconciliation diagnostics",
        status="Planned",
        description=(
            "Side-by-side comparison of raw and reconciled forecasts with coherence "
            "error metrics (sum-of-weekly vs monthly), using the mint_shrink method."
        ),
        expected_inputs=[
            "monthly_prophet_forecast_latest.parquet",
            "weekly_forecast_raw.parquet",
            "Reconciliation weights (reconciliation pipeline output)",
        ],
        planned_outputs=[
            "Coherence error table per month",
            "Before/after reconciliation chart",
        ],
    )
