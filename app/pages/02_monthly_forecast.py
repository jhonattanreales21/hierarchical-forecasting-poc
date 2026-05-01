import streamlit as st

from shared.viz import plot_forecast
from ui.page_blocks.monthly_blocks import (
    render_executive_forecast_summary,
    render_forecast_chart_panel,
    render_future_forecast_table,
    render_horizon_selector,
    render_monthly_data_status,
    render_monthly_kpi_summary,
    render_monthly_page_header,
    render_monthly_section_gap,
)
from ui.styles import apply_global_styles
from utils.data_loaders import (
    load_champion_metadata,
    load_champion_test_forecast,
    load_inference_metadata,
    load_monthly_forecast,
    load_monthly_modeling_data,
)
from utils.paths import CHAMPION_META

apply_global_styles()
render_monthly_page_header()

# ---------------------------------------------------------------------------
# Guard: champion pipeline must have run
# ---------------------------------------------------------------------------
if not CHAMPION_META.exists():
    from ui.components import render_empty_state

    render_empty_state(
        title="Champion Model Not Found",
        message=(
            "No champion model metadata was found. "
            "Run the monthly training pipeline first: "
            "`uv run kedro run --pipeline train_monthly`"
        ),
        status="warning",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------------
meta = load_champion_metadata()
inference_meta = load_inference_metadata()

champion_id: str = meta.get("champion_id", "")
test_metrics: dict = meta.get("test_metrics", {})

# ---------------------------------------------------------------------------
# Horizon selector — rendered here; KPI placeholder filled below
# ---------------------------------------------------------------------------
kpi_placeholder = st.container()

horizon_months = render_horizon_selector()

with kpi_placeholder:
    render_monthly_kpi_summary(meta, inference_meta, horizon_months)
render_monthly_section_gap()

# ---------------------------------------------------------------------------
# Load forecast data for selected horizon
# ---------------------------------------------------------------------------
actuals = load_monthly_modeling_data()

raw_test_fc = load_champion_test_forecast()
test_fc = (
    raw_test_fc[raw_test_fc["candidate_id"] == champion_id]
    .copy()
    .sort_values("ds")
    .reset_index(drop=True)
    if not raw_test_fc.empty
    else raw_test_fc
)

future_fc = load_monthly_forecast(horizon_months)

# ---------------------------------------------------------------------------
# Forecast chart
# ---------------------------------------------------------------------------
fig = plot_forecast(
    actuals=actuals,
    test_forecast=test_fc,
    future_forecast=future_fc,
    title="Monthly Demand Forecast — Prophet Champion",
    champion_id=champion_id,
    test_mape=test_metrics.get("mape"),
)
render_forecast_chart_panel(fig, meta, horizon_months)
render_monthly_section_gap()

# ---------------------------------------------------------------------------
# Executive interpretation
# ---------------------------------------------------------------------------
render_executive_forecast_summary(meta, inference_meta, horizon_months, future_fc)
render_monthly_section_gap()

# ---------------------------------------------------------------------------
# Forecast table + download
# ---------------------------------------------------------------------------
render_future_forecast_table(future_fc, horizon_months)
render_monthly_section_gap()

# ---------------------------------------------------------------------------
# Data availability status footer
# ---------------------------------------------------------------------------
render_monthly_data_status()
