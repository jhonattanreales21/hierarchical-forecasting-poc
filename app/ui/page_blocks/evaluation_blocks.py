"""Page-specific UI blocks for the Evaluation Report page (05_Evaluation_Report.py).

All functions receive already-loaded data and have no knowledge of artifact paths.
They are model-family agnostic: the production champion may be Prophet, SARIMAX,
or CatBoost, and family champions are shown separately from the production
champion.
"""

import pandas as pd
import streamlit as st

from ui.components import (
    render_kpi_card,
    render_section_header,
    render_success_banner,
    render_warning_banner,
)
from utils.champion import family_label
from utils.formatting import format_date, format_metric, format_optional, format_percentage

_MONTHLY_SELECTION_CMD = "uv run kedro run --pipeline monthly_model_selection"


def render_evaluation_summary(identity: dict, business_flag: bool) -> None:
    """Render the production champion evaluation summary with KPI cards.

    Args:
        identity: Normalized champion identity from ``extract_champion_identity``.
        business_flag: Whether the business precision target was met.
    """
    test_m = identity.get("test_metrics", {})
    champion_id = identity.get("champion_id")
    family = family_label(identity.get("model_family"))
    precision = test_m.get("forecast_precision")
    precision_threshold = 0.85

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
        f"Production Champion: {family} ({format_optional(champion_id)})",
        description="Held-out test performance of the model selected across all families.",
    )

    wape = test_m.get("wape")
    mase = test_m.get("mase")
    rmse = test_m.get("rmse")
    bias = test_m.get("bias")
    selection_metric = identity.get("selection_metric")
    selection_value = identity.get("selection_metric_value")

    cols = st.columns(4)
    with cols[0]:
        render_kpi_card(
            label="Champion Family",
            value=family,
            help_text=f"ID: {format_optional(champion_id)}",
        )
    with cols[1]:
        render_kpi_card(
            label="Test WAPE",
            value=format_percentage(wape) if wape is not None else "N/A",
            status="success" if (wape is not None and wape < 0.15) else "warning",
            help_text="Primary metric: weighted absolute % error",
        )
    with cols[2]:
        render_kpi_card(
            label="Forecast Precision",
            value=format_percentage(precision) if precision is not None else "N/A",
            help_text="1 − WAPE  ·  business target ≥ 85%",
            status="success" if business_flag else "warning",
        )
    with cols[3]:
        render_kpi_card(
            label="Test MASE",
            value=f"{mase:.3f}" if mase is not None else "N/A",
            help_text="MASE < 1: better than seasonal naïve",
            status="success" if (mase is not None and mase < 1.0) else "warning",
        )

    cols2 = st.columns(4)
    with cols2[0]:
        render_kpi_card(
            label="Test RMSE",
            value=(
                format_metric(rmse, decimals=1, suffix=" units")
                if rmse is not None
                else "N/A"
            ),
        )
    with cols2[1]:
        render_kpi_card(
            label="Test Bias",
            value=format_percentage(bias) if bias is not None else "N/A",
            help_text="Relative forecast bias (negative = under-forecast)",
        )
    with cols2[2]:
        render_kpi_card(
            label="Selection Metric",
            value=str(selection_metric).upper() if selection_metric else "N/A",
        )
    with cols2[3]:
        render_kpi_card(
            label="Selected Value",
            value=(
                format_percentage(selection_value)
                if selection_value is not None
                else "N/A"
            ),
        )

    test_period = identity.get("test_period", {}) or {}
    st.caption(
        f"Test window: {test_period.get('start_date', 'N/A')} → "
        f"{test_period.get('end_date', 'N/A')}  |  "
        f"Training cutoff: {format_date(identity.get('training_cutoff', ''))}  |  "
        f"Selection metric: {str(selection_metric).upper() if selection_metric else 'N/A'}"
    )


def render_family_champion_comparison(
    family_df: pd.DataFrame,
    production_family: str | None = None,
) -> None:
    """Render the per-family champion comparison table.

    Each row is the best candidate within a model family. The production champion
    family is highlighted. This is distinct from the single production champion.

    Args:
        family_df: Family champion summary DataFrame. Empty triggers a note.
        production_family: Family of the production champion, marked with a star.
    """
    render_section_header(
        "Family Champions",
        description=(
            "Best candidate within each eligible model family on the held-out test "
            "window. The production champion (★) is selected from these by the primary metric."
        ),
    )

    if family_df is None or family_df.empty:
        st.info(
            f"Family champion summary not available. Run `{_MONTHLY_SELECTION_CMD}`."
        )
        return

    cols = ["family", "family_champion_id", "wape", "mase", "rmse", "bias"]
    available = [c for c in cols if c in family_df.columns]
    display = family_df[available].copy()
    if "wape" in display.columns:
        display = display.sort_values("wape").reset_index(drop=True)

    if production_family and "family" in display.columns:
        display.insert(
            0,
            "champion",
            display["family"].map(
                lambda f: "★"
                if str(f).lower() == str(production_family).lower()
                else ""
            ),
        )

    for col in ["wape", "bias"]:
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
            "champion": "Champion",
            "family": "Family",
            "family_champion_id": "Candidate",
            "wape": "WAPE",
            "mase": "MASE",
            "rmse": "RMSE",
            "bias": "Bias",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("★ = current production champion family. Ranked by test WAPE (lower is better).")


def render_production_selection_summary(sel_df: pd.DataFrame) -> None:
    """Render the production champion selection provenance.

    Args:
        sel_df: Single-row production selection summary DataFrame.
    """
    render_section_header(
        "Selection Provenance",
        description="How the production champion was chosen across eligible families.",
    )

    if sel_df is None or sel_df.empty:
        st.info(f"Selection summary not available. Run `{_MONTHLY_SELECTION_CMD}`.")
        return

    row = sel_df.iloc[0]
    cols = st.columns(4)
    with cols[0]:
        render_kpi_card(
            label="Eligible Families",
            value=str(row.get("candidate_count", "N/A")),
            help_text="Candidates evaluated",
        )
    with cols[1]:
        render_kpi_card(
            label="Family Champions",
            value=str(row.get("family_champion_count", "N/A")),
        )
    with cols[2]:
        render_kpi_card(
            label="Primary Metric",
            value=str(row.get("primary_metric", "N/A")).upper(),
        )
    with cols[3]:
        render_kpi_card(
            label="Selected On",
            value=format_date(row.get("selection_timestamp", "")),
        )

    reason = row.get("selection_reason")
    if isinstance(reason, str) and reason:
        st.caption(reason)


def render_candidate_test_metrics_table(tm_df: pd.DataFrame) -> None:
    """Render the detailed per-candidate test metrics table across all families.

    Args:
        tm_df: Per-candidate test metrics DataFrame. Empty triggers a note.
    """
    render_section_header(
        "Detailed Candidate Test Metrics",
        description="Full per-candidate evaluation results including horizon-specific error.",
    )

    if tm_df is None or tm_df.empty:
        st.info(f"Candidate test metrics not available. Run `{_MONTHLY_SELECTION_CMD}`.")
        return

    display_cols = [
        "family",
        "candidate_id",
        "candidate_rank",
        "is_family_champion",
        "is_production_champion",
        "wape",
        "mase",
        "rmse",
        "bias",
        "test_m2_wape",
        "test_m3_wape",
    ]
    available = [c for c in display_cols if c in tm_df.columns]
    display = tm_df[available].copy()
    if "wape" in display.columns:
        display = display.sort_values("wape").reset_index(drop=True)

    for col in ["wape", "bias", "test_m2_wape", "test_m3_wape"]:
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
            "family": "Family",
            "candidate_id": "Candidate",
            "candidate_rank": "Rank",
            "is_family_champion": "Family champ",
            "is_production_champion": "Prod champ",
            "wape": "WAPE",
            "mase": "MASE",
            "rmse": "RMSE",
            "bias": "Bias",
            "test_m2_wape": "M+2 WAPE",
            "test_m3_wape": "M+3 WAPE",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Ranked by test WAPE. MASE < 1 beats the seasonal naïve baseline.")


def render_validation_notes(meta: dict) -> None:
    """Render collapsible validation protocol and champion model parameters.

    Args:
        meta: Champion model metadata dict.
    """
    with st.expander("Validation protocol & champion parameters"):
        st.markdown(
            "**Validation approach:** Time-based, leakage-safe splits. Hyperparameters "
            "are tuned on training data, shortlisted on validation, refit on train+validation, "
            "and the champion is chosen on a held-out test window — no random shuffling."
        )
        params = meta.get("hyperparameters") or meta.get("model_params") or {}
        if params:
            st.markdown("**Champion model hyperparameters:**")
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
