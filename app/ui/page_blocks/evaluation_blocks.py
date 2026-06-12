"""Page-specific UI blocks for the Evaluation Report page (05_Evaluation_Report.py).

All functions receive already-loaded data and have no knowledge of artifact paths.
They are model-family agnostic: the production champion may be Prophet, SARIMAX,
or CatBoost, and family champions are shown separately from the production
champion.
"""

import pandas as pd
import streamlit as st
from shared.viz import plot_feature_importance_bar

from ui.components import (
    render_kpi_card,
    render_section_header,
    render_success_banner,
    render_warning_banner,
)
from utils.champion import family_label
from utils.formatting import format_date, format_metric, format_optional, format_percentage

_MONTHLY_SELECTION_CMD = "uv run kedro run --pipeline monthly_model_selection"

# How each family's driver-importance statistic should be labelled in the UI. The
# values are NOT comparable across families, so the axis label states the statistic.
_IMPORTANCE_TYPE_LABELS: dict[str, dict[str, str]] = {
    "mean_abs_shap": {
        "axis": "Mean |SHAP value|",
        "method": "SHAP values (TreeExplainer) on the full-history refit",
    },
    "mean_abs_contribution": {
        "axis": "Mean |centered contribution|",
        "method": "Prophet component & regressor contributions",
    },
    "abs_coefficient": {
        "axis": "|Coefficient|",
        "method": "SARIMAX exogenous coefficients",
    },
}


def render_evaluation_summary(identity: dict, business_flag: bool) -> None:
    """Render the production champion evaluation summary with KPI cards.

    Args:
        identity: Normalized champion identity from ``extract_champion_identity``.
        business_flag: Whether the business precision target was met.
    """
    evaluation_metrics = identity.get("evaluation_metrics", {})
    champion_id = identity.get("champion_id")
    family = family_label(identity.get("model_family"))
    precision = evaluation_metrics.get("forecast_precision")
    precision_threshold = 0.85

    if business_flag:
        render_success_banner(
            title="Business accuracy target met",
            message=(
                f"Forecast precision ≥ {format_percentage(precision_threshold)} "
                "on the rolling-origin evaluation."
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
        description="Rolling-origin performance of the model selected across all families.",
    )

    wmape = evaluation_metrics.get("wmape")
    mase = evaluation_metrics.get("mase")
    rmse = evaluation_metrics.get("rmse")
    bias = evaluation_metrics.get("bias")
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
            label="WMAPE",
            value=format_percentage(wmape) if wmape is not None else "N/A",
            status="success" if (wmape is not None and wmape < 0.15) else "warning",
            help_text="Primary metric: weighted mean absolute % error",
        )
    with cols[2]:
        render_kpi_card(
            label="Forecast Precision",
            value=format_percentage(precision) if precision is not None else "N/A",
            help_text="1 − WMAPE  ·  business target ≥ 85%",
            status="success" if business_flag else "warning",
        )
    with cols[3]:
        render_kpi_card(
            label="MASE",
            value=f"{mase:.3f}" if mase is not None else "N/A",
            help_text="MASE < 1: better than seasonal naïve",
            status="success" if (mase is not None and mase < 1.0) else "warning",
        )

    cols2 = st.columns(4)
    with cols2[0]:
        render_kpi_card(
            label="RMSE",
            value=(
                format_metric(rmse, decimals=1, suffix=" units")
                if rmse is not None
                else "N/A"
            ),
        )
    with cols2[1]:
        render_kpi_card(
            label="Bias",
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

    evaluation = identity.get("evaluation", {}) or {}
    st.caption(
        f"Evaluation mode: {evaluation.get('mode', 'rolling_origin')}  |  "
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
            "Best candidate within each eligible model family on rolling-origin "
            "window. The production champion (★) is selected from these by the primary metric."
        ),
    )

    if family_df is None or family_df.empty:
        st.info(
            f"Family champion summary not available. Run `{_MONTHLY_SELECTION_CMD}`."
        )
        return

    cols = ["family", "family_champion_id", "wmape", "mase", "rmse", "bias"]
    available = [c for c in cols if c in family_df.columns]
    display = family_df[available].copy()
    if "wmape" in display.columns:
        display = display.sort_values("wmape").reset_index(drop=True)

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

    for col in ["wmape", "bias"]:
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
            "wmape": "WMAPE",
            "mase": "MASE",
            "rmse": "RMSE",
            "bias": "Bias",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption("★ = current production champion family. Ranked by WMAPE_M3 (lower is better).")


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
            value=str(row.get("family_champion_count", "N/A")),
            help_text="Number of model families with at least one eligible candidate",
        )
    with cols[1]:
        render_kpi_card(
            label="Total Candidates",
            value=str(row.get("candidate_count", "N/A")),
            help_text="Total candidates evaluated across all families",
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


def render_candidate_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Render the detailed per-candidate rolling-origin metrics table across families.

    Args:
        metrics_df: Per-candidate rolling-origin metrics DataFrame. Empty triggers a note.
    """
    render_section_header(
        "Detailed Candidate Metrics",
        description="Full per-candidate evaluation results including horizon-specific error.",
    )

    if metrics_df is None or metrics_df.empty:
        st.info(f"Candidate metrics not available. Run `{_MONTHLY_SELECTION_CMD}`.")
        return

    display_cols = [
        "family",
        "candidate_id",
        "candidate_rank",
        "is_family_champion",
        "is_production_champion",
        "wmape",
        "mase",
        "rmse",
        "bias",
        "wmape_m2",
        "wmape_m3",
    ]
    available = [c for c in display_cols if c in metrics_df.columns]
    display = metrics_df[available].copy()
    if "wmape_m3" in display.columns:
        display = display.sort_values("wmape_m3").reset_index(drop=True)

    for col in ["wmape", "bias", "wmape_m2", "wmape_m3"]:
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
            "wmape": "WMAPE",
            "mase": "MASE",
            "rmse": "RMSE",
            "bias": "Bias",
            "wmape_m2": "M+2 WMAPE",
            "wmape_m3": "M+3 WMAPE",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption("Ranked by WMAPE_M3. MASE < 1 beats the seasonal naïve baseline.")


def render_validation_notes(meta: dict) -> None:
    """Render collapsible validation protocol and champion model parameters.

    Args:
        meta: Champion model metadata dict.
    """
    with st.expander("Validation protocol & champion parameters"):
        st.markdown(
            "**Validation approach:** rolling-origin backtesting with time-based, "
            "leakage-safe cycles. Tuning and champion selection use the same pooled "
            "rolling-origin metrics; no random shuffling."
        )
        params = meta.get("hyperparameters") or meta.get("model_params") or {}
        if params:
            st.markdown("**Champion model hyperparameters:**")
            params_df = pd.DataFrame(
                [{"Parameter": k, "Value": str(v)} for k, v in params.items()]
            )
            st.dataframe(params_df, width="stretch", hide_index=True)

        active_regressors = meta.get("active_regressors", [])
        if active_regressors:
            st.caption(
                f"Active regressors ({len(active_regressors)}): "
                f"{', '.join(active_regressors)}"
            )


def render_champion_explainability(
    importance_df: pd.DataFrame,
    identity: dict,
) -> None:
    """Render the per-family champion driver-importance explanation.

    Lets the user pick a family champion and shows its top demand drivers as a single
    importance bar. The statistic differs per family (SHAP for CatBoost, contributions
    for Prophet, coefficients for SARIMAX), so the axis label and a caption state which
    method produced the values and warn that magnitudes are not comparable across
    families.

    Args:
        importance_df: Long-form importance table (``family``, ``feature``,
            ``importance``, ``importance_type``, ``champion_id``, ``rank``). Empty
            triggers a guidance note.
        identity: Normalized champion identity; its ``model_family`` sets the default
            selected family.
    """
    render_section_header(
        "Champion Explainability — Demand Drivers",
        description=(
            "What drives each family champion's prediction. CatBoost uses SHAP "
            "(TreeExplainer); Prophet uses component/regressor contributions; SARIMAX "
            "uses exogenous coefficients. Values are not comparable across families."
        ),
    )

    if importance_df is None or importance_df.empty or "family" not in importance_df.columns:
        st.info(
            "Driver-importance artifact not available "
            "(`monthly_family_champion_importance.parquet`). Run "
            f"`{_MONTHLY_SELECTION_CMD}` to generate SHAP/contribution/coefficient data."
        )
        return

    families = list(importance_df["family"].dropna().unique())
    if not families:
        st.info("No family-champion drivers were produced.")
        return

    production_family = str(identity.get("model_family") or "").lower()
    default_index = next(
        (i for i, f in enumerate(families) if str(f).lower() == production_family), 0
    )

    available_methods = (
        importance_df[["family", "importance_type"]]
        .drop_duplicates()
        .sort_values("family")
        .reset_index(drop=True)
    )
    cols = st.columns(3)
    with cols[0]:
        render_kpi_card(
            label="Families with drivers",
            value=str(len(families)),
        )
    with cols[1]:
        render_kpi_card(
            label="Driver rows",
            value=str(len(importance_df)),
        )
    with cols[2]:
        current_method = available_methods.loc[
            available_methods["family"].astype(str).str.lower() == production_family,
            "importance_type",
        ]
        render_kpi_card(
            label="Production method",
            value=current_method.iloc[0] if not current_method.empty else "N/A",
        )

    selected_family = st.radio(
        "Family champion",
        options=families,
        index=default_index,
        format_func=family_label,
        horizontal=True,
        key="explainability_family",
    )

    subset = importance_df[importance_df["family"] == selected_family].copy()
    if subset.empty:
        st.info(f"No drivers available for the {family_label(selected_family)} champion.")
        return

    champion_id = (
        str(subset["champion_id"].iloc[0]) if "champion_id" in subset.columns else None
    )
    importance_type = (
        str(subset["importance_type"].iloc[0])
        if "importance_type" in subset.columns
        else ""
    )
    labels = _IMPORTANCE_TYPE_LABELS.get(
        importance_type, {"axis": "Importance", "method": importance_type or "driver importance"}
    )

    max_features = int(len(subset))
    top_n = max_features
    if max_features > 3:
        top_n = st.slider(
            "Number of drivers to show",
            min_value=3,
            max_value=max_features,
            value=min(15, max_features),
            key="explainability_top_n",
        )

    star = " ★" if str(selected_family).lower() == production_family else ""
    fig = plot_feature_importance_bar(
        subset,
        title=f"{family_label(selected_family)} champion{star}: {format_optional(champion_id)}",
        top_n=top_n,
        subtitle=labels["method"],
        x_axis_title=labels["axis"],
    )
    st.plotly_chart(fig, width="stretch")
    if "computed_at" in subset.columns:
        st.caption(f"Importance artifact computed at: {subset['computed_at'].iloc[0]}")
    st.caption(
        f"Method: {labels['method']}. ★ marks the production champion family. "
        "Importance magnitudes are only comparable within a single family."
    )

    # SARIMAX coefficients carry sign + significance worth surfacing.
    if importance_type == "abs_coefficient" and "coefficient" in subset.columns:
        detail_cols = [
            c for c in ["feature", "coefficient", "std_err", "pvalue"] if c in subset.columns
        ]
        with st.expander("Coefficient detail (sign & significance)"):
            detail = subset.sort_values("importance", ascending=False)[detail_cols].rename(
                columns={
                    "feature": "Driver",
                    "coefficient": "Coefficient",
                    "std_err": "Std. error",
                    "pvalue": "p-value",
                }
            )
            st.dataframe(detail, width="stretch", hide_index=True)
