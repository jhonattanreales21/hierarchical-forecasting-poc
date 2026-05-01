"""Page-specific UI blocks for the Project Overview page (01_project_overview.py).

Each function receives already-loaded data or static content and has no knowledge
of artifact paths or business logic beyond what is passed as arguments.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components import render_section_header


def render_artifact_status(artifact_checks: list[tuple[str, Path]]) -> None:
    """Render a pipeline artifact availability table.

    Args:
        artifact_checks: List of (display_name, path) tuples to check.
    """
    render_section_header(
        "Pipeline Artifacts",
        description="Kedro pipeline outputs required by this application.",
    )
    art_df = pd.DataFrame(
        [
            {
                "Artifact": name,
                "Status": "✅ Available" if path.exists() else "❌ Missing",
            }
            for name, path in artifact_checks
        ]
    )
    st.dataframe(art_df, use_container_width=True, hide_index=True)


def render_temporal_hierarchy_overview() -> None:
    """Render the temporal hierarchy section with a structured table."""
    render_section_header(
        "Temporal Hierarchy",
        description=(
            "The forecasting hierarchy is strictly temporal. Monthly is the primary "
            "decision layer; weekly is a secondary operational complement."
        ),
    )
    st.markdown("""
| Layer | Role | Status |
|-------|------|--------|
| **Monthly** | Primary analytical and decision layer — main stakeholder output | Active (MVP) |
| **Weekly** | Secondary enhancement — 14-week operational complement | Planned |
| **Daily** | Low-priority exploratory extension | Disabled |
        """)
    st.caption(
        "Primary coherence target: Monthly ↔ Weekly  ·  Reconciliation method: mint_shrink"
    )


def render_model_candidates_overview() -> None:
    """Render the model candidates comparison table."""
    render_section_header(
        "Model Candidates",
        description="Core model families evaluated in the monthly layer.",
    )
    st.markdown("""
| Model | Description | Primary Layer |
|-------|-------------|---------------|
| **SARIMAX** | Structured statistical baseline with seasonal and exogenous components | Monthly |
| **Prophet** | Existing benchmark; handles seasonality and trend changes robustly | Monthly & Weekly |
| **CatBoost** | Main tabular candidate with full exogenous variable support | Monthly & Weekly |
| **N-HiTS** | Optional neural benchmark (Nixtla NeuralForecast) — exploratory only | Monthly (optional) |
        """)


def render_granularity_levels() -> None:
    """Render the forecast horizons and granularity summary."""
    render_section_header(
        "Forecast Horizons",
        description="Supported granularities and forecast horizon lengths per layer.",
    )
    st.markdown("""
| Granularity | Horizons | Notes |
|-------------|----------|-------|
| **Monthly** | 3 months, 6 months, 12 months | Primary layer — active |
| **Weekly** | 4 weeks, 9 weeks, 14 weeks | Secondary layer — planned |
| **Daily** | N/A | Disabled by parameter |
        """)
