import streamlit as st

from ui.components import (
    render_kpi_card,
    render_page_header,
    render_section_header,
    render_success_banner,
    render_warning_banner,
)
from ui.page_blocks.evaluation_blocks import render_family_champion_comparison
from ui.page_blocks.overview_blocks import (
    render_artifact_status,
    render_granularity_levels,
    render_model_candidates_overview,
    render_temporal_hierarchy_overview,
)
from ui.styles import apply_global_styles
from utils.champion import extract_champion_identity, family_label
from utils.data_loaders import (
    load_champion_metadata,
    load_family_champion_summary,
    load_inference_metadata,
    load_model_selection_summary,
)
from utils.formatting import (
    format_date,
    format_metric,
    format_optional,
    format_percentage,
)
from utils.paths import (
    CHAMPION_META,
    FAMILY_CHAMPION_SUMMARY,
    FORECAST_LATEST,
    INFERENCE_META,
    SELECTION_SUMMARY,
    TEST_METRICS,
    forecast_parquet,
)

apply_global_styles()

render_page_header(
    title="Project Overview",
    subtitle="Live production-champion status, pipeline artifacts, and PoC scope.",
    eyebrow="Hierarchical Demand Forecasting",
)

# ---------------------------------------------------------------------------
# Live champion status (data-driven section)
# ---------------------------------------------------------------------------
render_section_header(
    "Production Champion — Current Status",
    description=(
        "Read at runtime from the latest model-selection artifacts. The winning family "
        "is chosen from a time-aware competition among Prophet, SARIMAX, and CatBoost, "
        "and may change after a new pipeline run."
    ),
)

if not CHAMPION_META.exists():
    render_warning_banner(
        title="No champion model found",
        message=(
            "Run the monthly model competition and inference first "
            "(`uv run kedro run --pipeline monthly_forecast_e2e`)."
        ),
    )
else:
    meta = load_champion_metadata()
    inf = load_inference_metadata()
    selection_summary = load_model_selection_summary()
    identity = extract_champion_identity(meta, inf, selection_summary)

    test_m: dict = identity.get("test_metrics", {})
    wape = test_m.get("wape")
    rmse = test_m.get("rmse")
    precision = test_m.get("forecast_precision")
    business_flag = wape is not None and wape <= 0.15

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

    # --- KPI row 1: champion identity and accuracy ---
    cols = st.columns(4)
    with cols[0]:
        render_kpi_card(
            label="Champion Family",
            value=family_label(identity.get("model_family")),
            help_text=f"ID: {format_optional(identity.get('champion_id'))}",
        )
    with cols[1]:
        render_kpi_card(
            label="Test WAPE",
            value=format_percentage(wape) if wape is not None else "N/A",
            status="success" if (wape is not None and wape < 0.15) else "warning",
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
            help_text="1 − WAPE  ·  target ≥ 85%",
            status="success" if business_flag else "warning",
        )

    # --- KPI row 2: pipeline provenance ---
    supported = identity.get("supported_horizons") or []
    horizons_available = ", ".join(f"{h}m" for h in supported) if supported else "None"
    generated_at = identity.get("forecast_generated_at")
    last_run_date = format_date(generated_at) if generated_at else "N/A"
    selection_metric = identity.get("selection_metric")

    cols2 = st.columns(4)
    with cols2[0]:
        render_kpi_card(
            label="Selected On",
            value=format_date(identity.get("selected_at", "")),
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
            value=str(selection_metric).upper() if selection_metric else "N/A",
        )

    # --- Family champion comparison (production vs per-family) ---
    render_family_champion_comparison(
        load_family_champion_summary(),
        production_family=identity.get("model_family"),
    )

    # --- Artifact status ---
    artifact_checks = [
        ("Champion metadata", CHAMPION_META),
        ("Selection summary", SELECTION_SUMMARY),
        ("Family champion summary", FAMILY_CHAMPION_SUMMARY),
        ("Candidate test metrics", TEST_METRICS),
        ("Inference metadata", INFERENCE_META),
        ("Forecast 3m", forecast_parquet(3)),
        ("Forecast 6m", forecast_parquet(6)),
        ("Forecast latest", FORECAST_LATEST),
    ]
    render_artifact_status(artifact_checks)

    test_w = identity.get("test_period", {}) or {}
    refit = identity.get("refit", {}) or {}
    st.caption(
        f"Test window: {test_w.get('start_date', 'N/A')} → {test_w.get('end_date', 'N/A')}  |  "
        f"Refit: {refit.get('start_date', 'N/A')} → {refit.get('end_date', 'N/A')}  |  "
        f"Training cutoff: {format_date(identity.get('training_cutoff', ''))}"
    )

# ---------------------------------------------------------------------------
# Static project information
# ---------------------------------------------------------------------------
render_temporal_hierarchy_overview()
render_model_candidates_overview()
render_granularity_levels()
