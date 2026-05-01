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

    mape = test_m.get("mape")
    rmse = test_m.get("rmse")
    wmape = test_m.get("wmape")
    h2_mape = test_m.get("horizon_2_mape")
    h3_mape = test_m.get("horizon_3_mape")
    val_mape = val_m.get("mape")

    cols = st.columns(4)
    with cols[0]:
        render_kpi_card(
            label="Test MAPE",
            value=format_percentage(mape) if mape is not None else "N/A",
            status="success" if (mape is not None and mape < 0.15) else "warning",
        )
    with cols[1]:
        render_kpi_card(
            label="Test RMSE",
            value=format_metric(rmse, decimals=1, suffix=" units") if rmse is not None else "N/A",
        )
    with cols[2]:
        render_kpi_card(
            label="Test wMAPE",
            value=format_percentage(wmape) if wmape is not None else "N/A",
        )
    with cols[3]:
        render_kpi_card(
            label="Forecast Precision",
            value=format_percentage(precision) if precision is not None else "N/A",
            help_text="1 − MAPE  ·  business target ≥ 85%",
            status="success" if business_flag else "warning",
        )

    cols2 = st.columns(4)
    with cols2[0]:
        render_kpi_card(
            label="H+2 MAPE",
            value=format_percentage(h2_mape) if h2_mape is not None else "N/A",
            help_text="2-month ahead error",
        )
    with cols2[1]:
        render_kpi_card(
            label="H+3 MAPE",
            value=format_percentage(h3_mape) if h3_mape is not None else "N/A",
            help_text="3-month ahead error",
        )
    with cols2[2]:
        render_kpi_card(
            label="Val MAPE",
            value=format_percentage(val_mape) if val_mape is not None else "N/A",
            help_text="Validation set MAPE",
        )
    with cols2[3]:
        render_kpi_card(label="Champion ID", value=champion_id)

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
        description="All evaluated candidates ranked by test MAPE on the held-out window.",
    )

    if sel_df.empty:
        st.info("Selection summary not available. Run the `model_selection` pipeline.")
        return

    display_cols = [
        "candidate_id", "is_champion", "validation_rank", "test_rank",
        "mape", "rmse", "wmape", "forecast_precision",
        "horizon_2_mape", "horizon_3_mape", "business_success_flag",
    ]
    available = [c for c in display_cols if c in sel_df.columns]
    display = sel_df[available].copy()

    for col in ["mape", "wmape", "forecast_precision", "horizon_2_mape", "horizon_3_mape"]:
        if col in display.columns:
            display[col] = display[col].map(format_percentage)
    if "rmse" in display.columns:
        display["rmse"] = display["rmse"].map(lambda v: format_metric(v, decimals=1))

    display = display.rename(
        columns={
            "candidate_id": "Candidate",
            "is_champion": "Champion",
            "validation_rank": "Val rank",
            "test_rank": "Test rank",
            "mape": "MAPE",
            "rmse": "RMSE",
            "wmape": "wMAPE",
            "forecast_precision": "Precision",
            "horizon_2_mape": "H+2 MAPE",
            "horizon_3_mape": "H+3 MAPE",
            "business_success_flag": "Target met",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Accuracy target: ≥ 85% forecast precision.")


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
        "candidate_id", "status", "mae", "rmse", "mape", "wmape",
        "forecast_precision", "horizon_2_mape", "horizon_3_mape",
        "business_success_flag", "test_start_date", "test_end_date", "test_rows",
    ]
    available = [c for c in display_cols if c in tm_df.columns]
    display = tm_df[available].copy()

    for col in ["mape", "wmape", "forecast_precision", "horizon_2_mape", "horizon_3_mape"]:
        if col in display.columns:
            display[col] = display[col].map(format_percentage)
    for col in ["mae", "rmse"]:
        if col in display.columns:
            display[col] = display[col].map(lambda v: format_metric(v, decimals=2))

    display = display.rename(
        columns={
            "candidate_id": "Candidate",
            "status": "Status",
            "mae": "MAE",
            "rmse": "RMSE",
            "mape": "MAPE",
            "wmape": "wMAPE",
            "forecast_precision": "Precision",
            "horizon_2_mape": "H+2 MAPE",
            "horizon_3_mape": "H+3 MAPE",
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
