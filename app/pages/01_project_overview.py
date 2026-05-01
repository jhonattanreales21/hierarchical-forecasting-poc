import streamlit as st

from ui.components import (
    render_kpi_card,
    render_page_header,
    render_section_header,
    render_success_banner,
    render_warning_banner,
)
from ui.page_blocks.overview_blocks import (
    render_artifact_status,
    render_granularity_levels,
    render_model_candidates_overview,
    render_temporal_hierarchy_overview,
)
from ui.styles import apply_global_styles
from utils.data_loaders import load_champion_metadata, load_inference_metadata
from utils.formatting import (
    format_date,
    format_metric,
    format_optional,
    format_percentage,
)
from utils.paths import (
    CHAMPION_META,
    FORECAST_LATEST,
    SELECTION_FORECAST,
    SELECTION_SUMMARY,
    TEST_METRICS,
    forecast_parquet,
)

apply_global_styles()

render_page_header(
    title="Project Overview",
    subtitle="Live champion status, pipeline artifacts, and PoC scope.",
    eyebrow="Hierarchical Demand Forecasting",
)

# ---------------------------------------------------------------------------
# Live champion status (data-driven section)
# ---------------------------------------------------------------------------
render_section_header(
    "Champion Model — Current Status",
    description="Real-time status pulled from the latest pipeline run.",
)

if not CHAMPION_META.exists():
    render_warning_banner(
        title="No champion model found",
        message=(
            "Run the full Kedro pipeline first "
            "(`uv run kedro run --pipeline train_monthly`)."
        ),
    )
else:
    meta = load_champion_metadata()
    inf = load_inference_metadata()

    test_m: dict = meta.get("test_metrics", {})
    business_flag: bool = meta.get("business_success_flag", False)
    mape = test_m.get("mape")
    rmse = test_m.get("rmse")
    precision = test_m.get("forecast_precision")

    if business_flag:
        render_success_banner(
            title="Business target met",
            message="Forecast precision ≥ 85% on the held-out test window.",
        )
    else:
        render_warning_banner(
            title="Business target not yet met",
            message=(
                f"Current precision: {format_percentage(precision)} — target: ≥ 85%."
            ),
        )

    # --- KPI row 1: model identity and accuracy ---
    cols = st.columns(4)
    with cols[0]:
        render_kpi_card(
            label="Champion Model",
            value=format_optional(meta.get("champion_id")),
            help_text="Current monthly forecasting champion",
        )
    with cols[1]:
        render_kpi_card(
            label="Test MAPE",
            value=format_percentage(mape) if mape is not None else "N/A",
            status="success" if (mape is not None and mape < 0.15) else "warning",
        )
    with cols[2]:
        render_kpi_card(
            label="Test RMSE",
            value=(
                format_metric(rmse, decimals=1, suffix=" units")
                if rmse is not None
                else "N/A"
            ),
        )
    with cols[3]:
        render_kpi_card(
            label="Forecast Precision",
            value=format_percentage(precision) if precision is not None else "N/A",
            help_text="1 − MAPE  ·  target ≥ 85%",
            status="success" if business_flag else "warning",
        )

    # --- KPI row 2: pipeline provenance ---
    horizons_available = "None"
    last_run_date = "N/A"
    if inf:
        horizons = inf.get("horizons", {})
        if horizons:
            horizons_available = ", ".join(f"{h}m" for h in horizons.keys())
        created_at = inf.get("forecast_created_at", "")
        if created_at:
            last_run_date = format_date(created_at)

    cols2 = st.columns(4)
    with cols2[0]:
        render_kpi_card(
            label="Selected On",
            value=format_date(meta.get("selected_at", "")),
        )
    with cols2[1]:
        render_kpi_card(
            label="Horizons Available",
            value=horizons_available,
        )
    with cols2[2]:
        render_kpi_card(
            label="Last Forecast Run",
            value=last_run_date,
        )
    with cols2[3]:
        render_kpi_card(
            label="Selection Metric",
            value=meta.get("selection_metric", "N/A").upper(),
        )

    # --- Artifact status ---
    artifact_checks = [
        ("Champion metadata", CHAMPION_META),
        ("Test forecast", SELECTION_FORECAST),
        ("Selection summary", SELECTION_SUMMARY),
        ("Test metrics", TEST_METRICS),
        ("Forecast 3m", forecast_parquet(3)),
        ("Forecast 6m", forecast_parquet(6)),
        ("Forecast latest", FORECAST_LATEST),
    ]
    render_artifact_status(artifact_checks)

    train_w = meta.get("train_window", {})
    test_w = meta.get("test_window", {})
    st.caption(
        f"Train window: {train_w.get('start_date', 'N/A')} → {train_w.get('end_date', 'N/A')}  |  "
        f"Test window: {test_w.get('start_date', 'N/A')} → {test_w.get('end_date', 'N/A')}"
    )

# ---------------------------------------------------------------------------
# Static project information
# ---------------------------------------------------------------------------
render_temporal_hierarchy_overview()
render_model_candidates_overview()
render_granularity_levels()
