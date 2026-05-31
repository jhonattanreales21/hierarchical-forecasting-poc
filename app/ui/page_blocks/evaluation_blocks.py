"""Page-specific UI blocks for the Evaluation Report page (05_evaluation_report.py).

All functions receive already-loaded data and have no knowledge of artifact paths.
The page script handles all data loading and calls these render functions.
"""

import pandas as pd
import streamlit as st

from ui.components import (
    render_kpi_card,
    render_section_header,
    render_success_banner,
    render_warning_banner,
)
from utils.formatting import format_metric, format_percentage


def render_evaluation_summary(
    meta: dict,
    test_m: dict,
    val_m: dict,
    business_flag: bool,
) -> None:
    """Render the champion model evaluation summary with KPI cards.

    Args:
        meta: Champion model metadata dict.
        test_m: Test metrics sub-dict from metadata.
        val_m: Validation metrics sub-dict from metadata.
        business_flag: Whether the business precision target was met.
    """
    champion_id = meta.get("champion_id", "N/A")
    precision = test_m.get("forecast_precision")
    precision_threshold = meta.get("business_success_precision_threshold", 0.85)

    if business_flag:
        render_success_banner(
            title="Business accuracy target met",
            message=(
                f"Forecast precision ≥ {format_percentage(precision_threshold)} "
                "on the held-out test window."
            ),
        )
    else:
        render_warning_banner(
            title="Business accuracy target not yet met",
            message=(
                f"Current precision: {format_percentage(precision)} — "
                f"target: ≥ {format_percentage(precision_threshold)}."
            ),
        )

    render_section_header(
        f"Champion: {champion_id}",
        description="Key performance metrics on the held-out test window.",
    )

    wape = test_m.get("wape")
    mase = test_m.get("mase")
    rmse = test_m.get("rmse")
    mape = test_m.get("mape")
    h2_wape = test_m.get("horizon_2_wape")
    h3_wape = test_m.get("horizon_3_wape")
    val_wape = val_m.get("wape")

    cols = st.columns(4)
    with cols[0]:
        render_kpi_card(
            label="Test WAPE",
            value=format_percentage(wape) if wape is not None else "N/A",
            status="success" if (wape is not None and wape < 0.15) else "warning",
            help_text="Primary metric: weighted absolute % error",
        )
    with cols[1]:
        render_kpi_card(
            label="Test MASE",
            value=f"{mase:.3f}" if mase is not None else "N/A",
            help_text="MASE < 1: better than seasonal naïve",
            status="success" if (mase is not None and mase < 1.0) else "warning",
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
            help_text="1 − WAPE  ·  business target ≥ 85%",
            status="success" if business_flag else "warning",
        )

    cols2 = st.columns(4)
    with cols2[0]:
        render_kpi_card(
            label="H+2 WAPE",
            value=format_percentage(h2_wape) if h2_wape is not None else "N/A",
            help_text="2-month ahead weighted error",
        )
    with cols2[1]:
        render_kpi_card(
            label="H+3 WAPE",
            value=format_percentage(h3_wape) if h3_wape is not None else "N/A",
            help_text="3-month ahead weighted error",
        )
    with cols2[2]:
        render_kpi_card(
            label="Val WAPE",
            value=format_percentage(val_wape) if val_wape is not None else "N/A",
            help_text="Validation set WAPE",
        )
    with cols2[3]:
        render_kpi_card(label="Champion ID", value=champion_id)

    if mape is not None:
        with st.expander("Diagnostic metrics"):
            dcols = st.columns(3)
            with dcols[0]:
                render_kpi_card(label="MAPE (diag)", value=format_percentage(mape))
            with dcols[1]:
                render_kpi_card(label="wMAPE (diag)", value=format_percentage(test_m.get("wmape")))
            with dcols[2]:
                render_kpi_card(label="Val MAPE (diag)", value=format_percentage(val_m.get("mape")))

    train_w = meta.get("train_window", {})
    test_w = meta.get("test_window", {})
    st.caption(
        f"Test window: {test_w.get('start_date', 'N/A')} → {test_w.get('end_date', 'N/A')}  |  "
        f"Train window: {train_w.get('start_date', 'N/A')} → {train_w.get('end_date', 'N/A')}  |  "
        f"Selection metric: {meta.get('selection_metric', 'N/A').upper()}"
    )


def render_candidate_comparison(sel_df: pd.DataFrame) -> None:
    """Render the model candidate comparison table.

    Args:
        sel_df: Selection summary DataFrame. Empty DataFrame triggers a missing-data note.
    """
    render_section_header(
        "Candidate Comparison",
        description="All evaluated candidates ranked by test WAPE on the held-out window.",
    )

    if sel_df.empty:
        st.info("Selection summary not available. Run the `model_selection` pipeline.")
        return

    display_cols = [
        "candidate_id",
        "is_champion",
        "validation_rank",
        "test_rank",
        "wape",
        "mase",
        "rmse",
        "test_m2_wape",
        "test_m3_wape",
        "forecast_precision",
        "business_success_flag",
    ]
    available = [c for c in display_cols if c in sel_df.columns]
    display = sel_df[available].copy()

    for col in ["wape", "test_m2_wape", "test_m3_wape", "forecast_precision"]:
        if col in display.columns:
            display[col] = display[col].map(format_percentage)
    if "mase" in display.columns:
        display["mase"] = display["mase"].map(
            lambda v: f"{v:.3f}" if v is not None and not pd.isna(v) else "N/A"
        )
    if "rmse" in display.columns:
        display["rmse"] = display["rmse"].map(lambda v: format_metric(v, decimals=1))

    display = display.rename(
        columns={
            "candidate_id": "Candidate",
            "is_champion": "Champion",
            "validation_rank": "Val rank",
            "test_rank": "Test rank",
            "wape": "WAPE",
            "mase": "MASE",
            "rmse": "RMSE",
            "test_m2_wape": "M+2 WAPE",
            "test_m3_wape": "M+3 WAPE",
            "forecast_precision": "Precision",
            "business_success_flag": "Target met",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Accuracy target: ≥ 85% forecast precision. MASE < 1 beats seasonal naïve.")


def render_test_metrics_table(tm_df: pd.DataFrame) -> None:
    """Render the detailed per-candidate test metrics table.

    Args:
        tm_df: Test metrics DataFrame. Empty DataFrame triggers a missing-data note.
    """
    render_section_header(
        "Detailed Test Metrics",
        description="Full per-candidate evaluation results including horizon-specific error.",
    )

    if tm_df.empty:
        st.info("Test metrics not available. Run the `model_selection` pipeline.")
        return

    display_cols = [
        "candidate_id",
        "status",
        "wape",
        "mase",
        "mae",
        "rmse",
        "forecast_precision",
        "business_success_flag",
        "test_start_date",
        "test_end_date",
        "test_rows",
    ]
    available = [c for c in display_cols if c in tm_df.columns]
    display = tm_df[available].copy()

    for col in ["wape", "forecast_precision"]:
        if col in display.columns:
            display[col] = display[col].map(format_percentage)
    if "mase" in display.columns:
        display["mase"] = display["mase"].map(
            lambda v: f"{v:.3f}" if v is not None and not pd.isna(v) else "N/A"
        )
    for col in ["mae", "rmse"]:
        if col in display.columns:
            display[col] = display[col].map(lambda v: format_metric(v, decimals=2))

    display = display.rename(
        columns={
            "candidate_id": "Candidate",
            "status": "Status",
            "wape": "WAPE",
            "mase": "MASE",
            "mae": "MAE",
            "rmse": "RMSE",
            "forecast_precision": "Precision",
            "business_success_flag": "Target met",
            "test_start_date": "Test start",
            "test_end_date": "Test end",
            "test_rows": "Rows",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_validation_notes(meta: dict) -> None:
    """Render collapsible validation protocol and champion model parameters.

    Args:
        meta: Champion model metadata dict.
    """
    with st.expander("Validation protocol & champion parameters"):
        st.markdown(
            "**Validation approach:** Time-based rolling-origin splits. "
            "No random shuffling — all splits respect temporal order to prevent data leakage."
        )
        params = meta.get("model_params", {})
        if params:
            st.markdown("**Champion model parameters:**")
            params_df = pd.DataFrame(
                [{"Parameter": k, "Value": str(v)} for k, v in params.items()]
            )
            st.dataframe(params_df, use_container_width=True, hide_index=True)

        active_regressors = meta.get("active_regressors", [])
        if active_regressors:
            st.caption(
                f"Active regressors ({len(active_regressors)}): "
                f"{', '.join(active_regressors)}"
            )
