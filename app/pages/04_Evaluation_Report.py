import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

from ui.components import render_page_header, render_warning_banner
from ui.navigation import render_top_navbar
from ui.page_blocks.evaluation_blocks import (
    render_candidate_metrics_table,
    render_champion_explainability,
    render_evaluation_summary,
    render_family_champion_comparison,
    render_production_selection_summary,
    render_validation_notes,
)
from ui.styles import apply_global_styles
from utils.champion import extract_champion_identity
from utils.data_loaders import (
    load_champion_metadata,
    load_candidate_metrics,
    load_family_champion_importance,
    load_family_champion_summary,
    load_inference_metadata,
    load_model_selection_summary,
)
from utils.paths import CHAMPION_META

apply_global_styles()
render_top_navbar("Evaluation Report")

render_page_header(
    title="Evaluation Report",
    subtitle=(
        "Champion selection rationale and rolling-origin performance across "
        "Prophet, SARIMAX, and CatBoost."
    ),
    eyebrow="Model Evaluation",
)

if not CHAMPION_META.exists():
    render_warning_banner(
        title="No evaluation data found",
        message=(
            "Run the monthly model competition first "
            "(`uv run kedro run --pipeline monthly_model_selection`)."
        ),
    )
    st.stop()

meta = load_champion_metadata()
inference_meta = load_inference_metadata()
selection_summary = load_model_selection_summary()
identity = extract_champion_identity(meta, inference_meta, selection_summary)

wmape = identity.get("evaluation_metrics", {}).get("wmape")
business_flag = wmape is not None and wmape <= 0.15

render_evaluation_summary(identity, business_flag)

render_production_selection_summary(selection_summary)

render_family_champion_comparison(
    load_family_champion_summary(),
    production_family=identity.get("model_family"),
)

render_champion_explainability(load_family_champion_importance(), identity)

render_candidate_metrics_table(load_candidate_metrics())

render_validation_notes(meta)
