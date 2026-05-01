import pandas as pd
import streamlit as st

from ui.components import render_page_header, render_warning_banner
from ui.page_blocks.evaluation_blocks import (
    render_candidate_comparison,
    render_evaluation_summary,
    render_test_metrics_table,
    render_validation_notes,
)
from ui.styles import apply_global_styles
from utils.data_loaders import (
    load_champion_metadata,
    load_model_selection_summary,
    load_test_metrics,
)
from utils.paths import CHAMPION_META, SELECTION_SUMMARY, TEST_METRICS

apply_global_styles()

render_page_header(
    title="Evaluation Report",
    subtitle="Champion model selection rationale and held-out test performance.",
    eyebrow="Model Evaluation",
)

if not CHAMPION_META.exists():
    render_warning_banner(
        title="No evaluation data found",
        message=(
            "Run the full Kedro pipeline first "
            "(`uv run kedro run --pipeline model_selection`)."
        ),
    )
    st.stop()

meta = load_champion_metadata()
test_m: dict = meta.get("test_metrics", {})
val_m: dict = meta.get("validation_metrics", {})
business_flag: bool = meta.get("business_success_flag", False)

render_evaluation_summary(meta, test_m, val_m, business_flag)

sel_df = load_model_selection_summary() if SELECTION_SUMMARY.exists() else pd.DataFrame()
render_candidate_comparison(sel_df)

tm_df = load_test_metrics() if TEST_METRICS.exists() else pd.DataFrame()
render_test_metrics_table(tm_df)

render_validation_notes(meta)
