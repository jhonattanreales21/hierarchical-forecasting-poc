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
    render_monthly_provenance,
    render_monthly_section_gap,
)
from ui.styles import apply_global_styles
from utils.champion import extract_champion_identity, family_label, forecast_has_intervals
from utils.data_loaders import (
    load_champion_metadata,
    load_inference_metadata,
    load_legacy_test_forecast,
    load_model_selection_summary,
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
            "No champion model metadata was found. Run the monthly model competition "
            "and inference first: `uv run kedro run --pipeline monthly_forecast_e2e`"
        ),
        status="warning",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load metadata and build a model-family-agnostic champion identity
# ---------------------------------------------------------------------------
meta = load_champion_metadata()
inference_meta = load_inference_metadata()
selection_summary = load_model_selection_summary()
identity = extract_champion_identity(meta, inference_meta, selection_summary)

champion_id: str = identity.get("champion_id") or ""
test_metrics: dict = identity.get("test_metrics", {})

# ---------------------------------------------------------------------------
# Horizon selector — rendered here; KPI placeholder filled below
# ---------------------------------------------------------------------------
kpi_placeholder = st.container()

horizon_months = render_horizon_selector()

with kpi_placeholder:
    render_monthly_kpi_summary(identity, horizon_months)
render_monthly_section_gap()

# ---------------------------------------------------------------------------
# Load forecast data for selected horizon
# ---------------------------------------------------------------------------
actuals = load_monthly_modeling_data()

# Legacy Prophet test-period backtest, shown only as a held-out overlay when the
# current champion is the matching candidate. It never drives champion identity.
raw_test_fc = load_legacy_test_forecast()
test_fc = (
    raw_test_fc[raw_test_fc["candidate_id"] == champion_id]
    .copy()
    .sort_values("ds")
    .reset_index(drop=True)
    if not raw_test_fc.empty and "candidate_id" in raw_test_fc.columns
    else raw_test_fc.iloc[0:0]
)

future_fc = load_monthly_forecast(horizon_months)
has_intervals = forecast_has_intervals(future_fc, identity)

# ---------------------------------------------------------------------------
# Forecast chart
# ---------------------------------------------------------------------------
_family = identity.get("model_family")
chart_title = (
    f"Monthly Demand Forecast — {family_label(_family)} Champion"
    if _family
    else "Monthly Demand Forecast — Production Champion"
)
fig = plot_forecast(
    actuals=actuals,
    test_forecast=test_fc,
    future_forecast=future_fc,
    title=chart_title,
    champion_id=champion_id,
    test_mape=test_metrics.get("mape"),
    show_future_intervals=has_intervals,
)
render_forecast_chart_panel(fig, identity, horizon_months, has_intervals)
render_monthly_provenance(identity)
render_monthly_section_gap()

# ---------------------------------------------------------------------------
# Executive interpretation
# ---------------------------------------------------------------------------
render_executive_forecast_summary(identity, horizon_months, future_fc)
render_monthly_section_gap()

# ---------------------------------------------------------------------------
# Forecast table + download
# ---------------------------------------------------------------------------
render_future_forecast_table(future_fc, horizon_months, has_intervals)
render_monthly_section_gap()

# ---------------------------------------------------------------------------
# Data availability status footer
# ---------------------------------------------------------------------------
render_monthly_data_status()
