"""Monthly multi-family model selection nodes.

Selects one champion per model family from the pooled rolling-origin metrics
produced by training, then elects a single monthly production champion across
families.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .explainability import (
    IMPORTANCE_COLUMNS,
    assemble_family_importance_table,
    compute_catboost_shap_importance,
    compute_prophet_regressor_importance,
    compute_sarimax_coefficient_importance,
)

logger = logging.getLogger(__name__)


# ── Node 1 ────────────────────────────────────────────────────────────────────


_ROLLING_ORIGIN_METRIC_KEYS: tuple[str, ...] = (
    "wmape",
    "wmape_m1",
    "wmape_m2",
    "wmape_m3",
    "mase",
    "bias",
    "rmse",
)


def assemble_monthly_candidate_metrics(
    monthly_prophet_prechampion_configs: dict,
    monthly_sarimax_prechampion_configs: dict,
    params_monthly: dict,
    monthly_catboost_prechampion_configs: dict | None = None,
) -> pd.DataFrame:
    """Assemble the cross-family candidate metrics table from the pre-champions.

    Champions are selected directly from the pooled rolling-origin metrics already
    produced by each family tuner. One row per pre-champion candidate, with the
    WMAPE / per-horizon WMAPE / MASE / BIAS metric set plus the SARIMAX Ljung-Box
    eligibility flag.

    Args:
        monthly_prophet_prechampion_configs: Prophet ``prechampion_configs`` artifact.
        monthly_sarimax_prechampion_configs: SARIMAX ``prechampion_configs`` artifact.
        params_monthly: Contents of ``model_selection.monthly`` (reads
            ``active_families`` and ``require_all_active_families``).
        monthly_catboost_prechampion_configs: CatBoost ``prechampion_configs`` artifact
            (required only when ``catboost`` is in ``active_families``).

    Returns:
        DataFrame with one row per candidate across active families.

    Raises:
        ValueError: When an active family has no candidates while
            ``require_all_active_families`` is set, or when no candidates can be
            assembled at all.
    """
    active_families = [
        str(f) for f in params_monthly.get("active_families", ["prophet", "sarimax"])
    ]
    require_all = bool(params_monthly.get("require_all_active_families", True))

    # Per family: its prechampion artifact and the key holding the candidate list.
    family_sources: dict[str, tuple[dict, str]] = {
        "prophet": (monthly_prophet_prechampion_configs or {}, "prechampions"),
        "sarimax": (monthly_sarimax_prechampion_configs or {}, "candidates"),
        "catboost": (monthly_catboost_prechampion_configs or {}, "candidates"),
    }

    rows: list[dict] = []
    missing: list[str] = []
    for family in active_families:
        configs, list_key = family_sources.get(family, ({}, "candidates"))
        family_rows = _candidate_rows(configs, family, list_key)
        if not family_rows:
            missing.append(family)
            continue
        rows.extend(family_rows)

    if missing and require_all:
        raise ValueError(
            "require_all_active_families is set but these active families have no "
            f"prechampion candidates: {missing}"
        )
    for family in missing:
        logger.warning(
            "Active family '%s' has no prechampion candidates — skipped.", family
        )

    if not rows:
        raise ValueError(
            "No candidate metrics could be assembled from the pre-champion configs."
        )
    return pd.DataFrame(rows)


def _candidate_rows(
    prechampion_configs: dict, family: str, list_key: str
) -> list[dict]:
    """Build candidate metric rows from a family's prechampion_configs artifact."""
    entries = list((prechampion_configs or {}).get(list_key, []))
    rows: list[dict] = []
    for entry in entries:
        candidate_id = str(entry.get("candidate_id") or entry.get("trial_id"))
        metrics = dict(entry.get("rolling_origin_metrics", {}))
        row = {
            "family": family,
            "candidate_id": candidate_id,
            "granularity": "monthly",
            "candidate_rank": _safe_int(entry.get("rank")),
            "selection_stage": "rolling_origin",
            "ljung_box_pvalue": _safe_float(entry.get("ljung_box_pvalue")),
            "autocorrelation_excluded": bool(
                entry.get("autocorrelation_excluded", False)
            ),
            "ljung_box_cycle_index": _safe_int(entry.get("ljung_box_cycle_index")),
            "primary_metric": "wmape_m3",
            "primary_metric_value": _safe_float(metrics.get("wmape_m3")),
            "is_family_champion": False,
            "is_production_champion": False,
        }
        for key in _ROLLING_ORIGIN_METRIC_KEYS:
            row[key] = _safe_float(metrics.get(key))
        rows.append(row)
    return rows


def select_monthly_family_champions(
    monthly_candidate_metrics: pd.DataFrame,
    params_monthly: dict,
) -> pd.DataFrame:
    """Select the best rolling-origin candidate within each active model family.

    Ranks candidates by the configured primary metric (default: ``wmape_m3``) and
    applies tie-breakers in order. Exactly one family champion is
    selected per active family. For SARIMAX, candidates flagged by the Ljung-Box
    filter are excluded before ranking, with a fallback to the full
    set when none are eligible.

    Args:
        monthly_candidate_metrics: Output of ``assemble_monthly_candidate_metrics``.
        params_monthly: Contents of ``model_selection.monthly``; must contain
            ``primary_metric`` and ``tie_breakers``.

    Returns:
        DataFrame with one row per family champion (family, granularity,
        family_champion_id, family_champion_rank, the rolling-origin metric set,
        selection_reason, and artifact keys).

    Raises:
        ValueError: When the input metrics table is empty.
    """
    if monthly_candidate_metrics.empty:
        raise ValueError(
            "monthly_candidate_metrics is empty; cannot select family champions."
        )

    primary_metric: str = str(params_monthly.get("primary_metric", "wmape_m3"))
    tie_breakers: list[str] = list(
        params_monthly.get("tie_breakers", ["wmape_m2", "wmape_m1", "mase", "abs_bias"])
    )
    active_families: list[str] = sorted(
        monthly_candidate_metrics["family"].unique().tolist()
    )

    champion_rows: list[dict] = []
    for family in active_families:
        family_df = monthly_candidate_metrics[
            monthly_candidate_metrics["family"] == family
        ].copy()
        if family_df.empty:
            logger.warning("No candidates found for family '%s' — skipping.", family)
            continue

        # SARIMAX: exclude Ljung-Box-flagged candidates before ranking.
        if family == "sarimax" and "autocorrelation_excluded" in family_df.columns:
            eligible = family_df[~family_df["autocorrelation_excluded"].fillna(False)]
            if eligible.empty:
                logger.warning(
                    "All SARIMAX candidates were excluded by the Ljung-Box residual "
                    "autocorrelation filter — falling back to full candidate set."
                )
                eligible = family_df
            family_df = eligible

        family_df["abs_bias"] = family_df["bias"].abs()
        sort_cols = _build_sort_columns(primary_metric, tie_breakers, family_df.columns)
        ranked = family_df.sort_values(sort_cols, ascending=True, na_position="last")
        best = ranked.iloc[0]

        champion_rows.append(
            {
                "family": family,
                "granularity": "monthly",
                "family_champion_id": best["candidate_id"],
                "family_champion_rank": 1,
                **{k: _safe_float(best.get(k)) for k in _ROLLING_ORIGIN_METRIC_KEYS},
                "ljung_box_pvalue": _safe_float(best.get("ljung_box_pvalue")),
                "selection_reason": (
                    f"Best {primary_metric} among {len(family_df)} eligible {family} "
                    f"candidate(s) on the rolling-origin backtest"
                ),
                "model_artifact_key": f"monthly_{family}_candidate_models",
                "metadata_artifact_key": f"monthly_{family}_prechampion_configs",
            }
        )
        _m = {k: _safe_float(best.get(k)) for k in _ROLLING_ORIGIN_METRIC_KEYS}
        logger.info(
            "Family champion (%s): %s  wmape=%.4f  wmape_m1=%.4f  wmape_m2=%.4f  "
            "wmape_m3=%.4f  mase=%s  bias=%s  rmse=%s",
            family,
            best["candidate_id"],
            _m["wmape"] if _m["wmape"] is not None else float("nan"),
            _m["wmape_m1"] if _m["wmape_m1"] is not None else float("nan"),
            _m["wmape_m2"] if _m["wmape_m2"] is not None else float("nan"),
            _m["wmape_m3"] if _m["wmape_m3"] is not None else float("nan"),
            f"{_m['mase']:.4f}" if _m["mase"] is not None else "n/a",
            f"{_m['bias']:+.4f}" if _m["bias"] is not None else "n/a",
            f"{_m['rmse']:.2f}" if _m["rmse"] is not None else "n/a",
        )

    if not champion_rows:
        raise RuntimeError(
            "No family champions could be selected from the metrics table."
        )
    return pd.DataFrame(champion_rows)


# ── Node 2 ────────────────────────────────────────────────────────────────────


def select_monthly_production_champion(
    monthly_family_champion_summary: pd.DataFrame,
    monthly_candidate_metrics: pd.DataFrame,
    params_monthly: dict,
) -> pd.DataFrame:
    """Compare family champions and elect one monthly production champion.

    Only family champions (one per family) participate in the final comparison.
    The selection uses the same metric ordering as family champion selection:
    primary metric first, then tie-breakers in order.

    Args:
        monthly_family_champion_summary: Output of ``select_monthly_family_champions``.
        monthly_candidate_metrics: Full candidate metrics table for summary stats.
        params_monthly: Contents of ``model_selection.monthly``.

    Returns:
        Single-row DataFrame describing the production champion selection.

    Raises:
        ValueError: When the family champion summary is empty.
    """
    if monthly_family_champion_summary.empty:
        raise ValueError(
            "monthly_family_champion_summary is empty; cannot select production champion."
        )

    primary_metric: str = str(params_monthly.get("primary_metric", "wmape_m3"))
    tie_breakers: list[str] = list(
        params_monthly.get("tie_breakers", ["wmape_m2", "wmape_m1", "mase", "abs_bias"])
    )
    active_families: list[str] = list(
        params_monthly.get("active_families", ["prophet", "sarimax"])
    )

    champions_df = monthly_family_champion_summary.copy()
    champions_df["abs_bias"] = champions_df["bias"].abs()
    sort_cols = _build_sort_columns(primary_metric, tie_breakers, champions_df.columns)
    ranked = champions_df.sort_values(sort_cols, ascending=True, na_position="last")
    best = ranked.iloc[0]

    runner_up_info = ""
    if len(ranked) > 1:
        runner_up = ranked.iloc[1]
        gap = _safe_float(best.get(primary_metric))
        runner_up_val = _safe_float(runner_up.get(primary_metric))
        if gap is not None and runner_up_val is not None:
            runner_up_info = (
                f"; runner-up {runner_up['family']} "
                f"({primary_metric}={runner_up_val:.4f}, gap={runner_up_val - gap:.4f})"
            )

    selection_reason = (
        f"Best {primary_metric} family champion among {len(ranked)} eligible "
        f"families{runner_up_info}"
    )

    _log_production_champion_comparison(ranked, primary_metric, tie_breakers)

    summary_row = {
        "granularity": "monthly",
        "active_families": active_families,
        "production_champion_family": best["family"],
        "production_champion_id": best["family_champion_id"],
        "primary_metric": primary_metric,
        "primary_metric_value": _safe_float(best.get(primary_metric)),
        "tie_breakers": tie_breakers,
        "selection_timestamp": datetime.now(tz=UTC).isoformat(),
        "selection_reason": selection_reason,
        "candidate_count": len(monthly_candidate_metrics),
        "family_champion_count": len(ranked),
    }
    return pd.DataFrame([summary_row])


# ── Node 3 ────────────────────────────────────────────────────────────────────


def annotate_monthly_candidate_champion_flags(
    monthly_candidate_metrics: pd.DataFrame,
    monthly_family_champion_summary: pd.DataFrame,
    monthly_model_selection_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Mark family and production champions in the candidate metrics table.

    ``monthly_candidate_metrics`` is the app/reporting-facing audit table, so
    its champion flags must reflect the separately persisted family and production
    champion summaries.

    Args:
        monthly_candidate_metrics: Rolling-origin metrics for all candidates.
        monthly_family_champion_summary: One selected champion row per model family.
        monthly_model_selection_summary: One selected production champion row.

    Returns:
        Copy of the candidate metrics table with coherent ``is_family_champion`` and
        ``is_production_champion`` boolean flags.

    Raises:
        ValueError: When candidate metrics are empty, no family champions are
            available, or the production champion summary is empty.
    """
    if monthly_candidate_metrics.empty:
        raise ValueError(
            "monthly_candidate_metrics is empty; cannot annotate champion flags."
        )
    if monthly_family_champion_summary.empty:
        raise ValueError(
            "monthly_family_champion_summary is empty; cannot annotate champion flags."
        )
    if monthly_model_selection_summary.empty:
        raise ValueError(
            "monthly_model_selection_summary is empty; cannot annotate champion flags."
        )

    metrics_df = monthly_candidate_metrics.copy()
    family_champion_pairs = set(
        zip(
            monthly_family_champion_summary["family"].astype(str),
            monthly_family_champion_summary["family_champion_id"].astype(str),
            strict=True,
        )
    )

    production_row = monthly_model_selection_summary.iloc[0]
    production_pair = (
        str(production_row["production_champion_family"]),
        str(production_row["production_champion_id"]),
    )

    candidate_pairs = list(
        zip(
            metrics_df["family"].astype(str),
            metrics_df["candidate_id"].astype(str),
            strict=True,
        )
    )
    metrics_df["is_family_champion"] = [
        pair in family_champion_pairs for pair in candidate_pairs
    ]
    metrics_df["is_production_champion"] = [
        pair == production_pair for pair in candidate_pairs
    ]

    family_flag_count = int(metrics_df["is_family_champion"].sum())
    production_flag_count = int(metrics_df["is_production_champion"].sum())
    expected_family_count = len(family_champion_pairs)
    if family_flag_count != expected_family_count:
        raise ValueError(
            "Could not map all family champions back to candidate metrics: "
            f"expected {expected_family_count}, marked {family_flag_count}."
        )
    if production_flag_count != 1:
        raise ValueError(
            "Could not map exactly one production champion back to candidate metrics: "
            f"marked {production_flag_count}."
        )

    return metrics_df


# ── Node 4 ────────────────────────────────────────────────────────────────────


def build_monthly_champion_artifacts(  # noqa: PLR0913
    monthly_model_selection_summary: pd.DataFrame,
    monthly_family_champion_summary: pd.DataFrame,
    monthly_candidate_metrics: pd.DataFrame,
    monthly_prophet_candidate_models: dict,
    monthly_sarimax_candidate_models: dict,
    monthly_prophet_full_train: pd.DataFrame,
    monthly_sarimax_full_train: pd.DataFrame,
    monthly_sarimax_training_metadata: dict,
    params_monthly: dict,
    monthly_catboost_candidate_models: dict | None = None,
    monthly_catboost_full_train: pd.DataFrame | None = None,
    monthly_catboost_split_metadata: dict | None = None,
) -> tuple[Any, dict]:
    """Refit the production champion on full history and build its JSON metadata.

    Champion Protocol stage 5 (full-history refit): the elected champion
    configuration is refit on all available history (train + validation + test) so
    that production inference benefits from the full training window. Reported test
    metrics are unchanged — they come from the pre-refit selection stage. When
    ``refit_champion.enabled`` is false, the train-only candidate is returned as-is.

    The champion may belong to any active family (Prophet, SARIMAX, or CatBoost); the
    returned model artifact and metadata are family-aware and metadata-driven, so the
    inference layer never branches on the Python type of the model.

    Args:
        monthly_model_selection_summary: Single-row summary from
            ``select_monthly_production_champion``.
        monthly_family_champion_summary: Family champions table from
            ``select_monthly_family_champions``.
        monthly_candidate_metrics: Full candidate metrics table (rolling-origin metrics
            across all families, produced by ``assemble_monthly_candidate_metrics``).
        monthly_prophet_candidate_models: Dict mapping candidate_id → Prophet model.
        monthly_sarimax_candidate_models: Dict mapping trial_id → SARIMAX candidate
            entry dict (including the ``model`` key).
        monthly_prophet_full_train: Prophet-format full history (ds, y, regressors).
        monthly_sarimax_full_train: SARIMAX-format full history (date, target, exog).
        monthly_sarimax_training_metadata: SARIMAX training metadata (exogenous_columns).
        params_monthly: Contents of ``model_selection.monthly``.
        monthly_catboost_candidate_models: Dict mapping candidate_id → CatBoost
            candidate entry dict (``config``, ``model``, ``feature_columns``).
            Required only when the elected champion family is ``catboost``.
        monthly_catboost_full_train: CatBoost-format full history (precomputed
            features + target). Required only for a CatBoost champion refit.
        monthly_catboost_split_metadata: CatBoost split metadata (feature/categorical/
            target/date columns). Required only for a CatBoost champion.

    Returns:
        Two-element tuple:

        1. ``champion_monthly_model`` — the production champion model, refit on full
           history (Prophet model; SARIMAX candidate dict carrying the refit results
           under ``"model"``; or CatBoost candidate dict carrying the refit
           ``CatBoostRegressor`` under ``"model"`` plus ``"feature_columns"``).
        2. ``champion_monthly_metadata`` — JSON-serialisable dict describing the
           champion, metrics, test period, refit, and inference contract.

    Raises:
        ValueError: When the summary is empty or the champion family is unknown.
        RuntimeError: When the champion candidate ID cannot be resolved to a model.
    """
    if monthly_model_selection_summary.empty:
        raise ValueError("monthly_model_selection_summary is empty.")

    summary_row = monthly_model_selection_summary.iloc[0]
    production_family: str = str(summary_row["production_champion_family"])
    production_candidate_id: str = str(summary_row["production_champion_id"])

    candidate_model = _resolve_champion_model(
        production_family=production_family,
        production_candidate_id=production_candidate_id,
        prophet_candidate_models=monthly_prophet_candidate_models,
        sarimax_candidate_models=monthly_sarimax_candidate_models,
        catboost_candidate_models=monthly_catboost_candidate_models or {},
    )

    champion_model, inference_contract, refit_info = _build_production_champion_model(
        production_family=production_family,
        candidate_model=candidate_model,
        prophet_full_train=monthly_prophet_full_train,
        sarimax_full_train=monthly_sarimax_full_train,
        sarimax_training_metadata=monthly_sarimax_training_metadata,
        catboost_full_train=monthly_catboost_full_train,
        catboost_split_metadata=monthly_catboost_split_metadata or {},
        params_monthly=params_monthly,
    )

    champ_metrics = _extract_champion_metrics(
        monthly_candidate_metrics, production_family, production_candidate_id
    )

    family_champions: dict[str, dict] = {}
    for _, fc_row in monthly_family_champion_summary.iterrows():
        fam = str(fc_row["family"])
        family_champions[fam] = {
            "champion_id": str(fc_row["family_champion_id"]),
            "wmape_m3": _safe_float(fc_row.get("wmape_m3")),
            "wmape": _safe_float(fc_row.get("wmape")),
            "mase": _safe_float(fc_row.get("mase")),
            "bias": _safe_float(fc_row.get("bias")),
        }

    metadata: dict = {
        "granularity": "monthly",
        "model_family": production_family,
        "champion_id": production_candidate_id,
        "champion_level": "production",
        "active_regressors": list(inference_contract["active_regressors"]),
        "training_cutoff": refit_info.get("end_date"),
        "hyperparameters": _extract_champion_hyperparameters(
            production_family, champion_model, inference_contract
        ),
        "family_champions": family_champions,
        "selection": {
            "primary_metric": str(summary_row.get("primary_metric", "wmape_m3")),
            "direction": "minimize",
            "tie_breakers": list(
                params_monthly.get(
                    "tie_breakers", ["wmape_m2", "wmape_m1", "mase", "abs_bias"]
                )
            ),
            "selected_at": str(
                summary_row.get("selection_timestamp", datetime.now(tz=UTC).isoformat())
            ),
            "selection_reason": str(summary_row.get("selection_reason", "")),
        },
        "evaluation": {
            "mode": "rolling_origin",
            "primary_metric": "wmape_m3",
            "note": (
                "Champions selected directly on pooled rolling-origin metrics; "
                "no reserved out-of-sample window was used (optimistic-selection risk)."
            ),
        },
        "metrics": champ_metrics,
        "inference_contract": inference_contract,
        "refit": refit_info,
        "model_artifact": {
            "catalog_key": "champion_monthly_model",
            "source_candidate_key": f"monthly_{production_family}_candidate_models",
            "source_candidate_id": production_candidate_id,
        },
        "compatibility": {
            "legacy_prophet_champion_preserved": True,
            "inference_ready": True,
            "metadata_driven_inference": True,
        },
    }

    # CatBoost direct multi-horizon inference needs the feature contract at the
    # metadata top level so the inference adapter can dispatch correctly without
    # introspecting the model object.
    if production_family == "catboost":
        metadata["feature_columns"] = list(
            inference_contract.get("feature_columns", [])
        )
        metadata["categorical_features"] = list(
            inference_contract.get("categorical_features", [])
        )
        metadata["target_column"] = inference_contract.get("target_column")
        metadata["date_column"] = inference_contract.get("date_column")
        metadata["strategy"] = inference_contract.get(
            "strategy", "direct_multi_horizon"
        )
        metadata["max_forecast_horizon"] = inference_contract.get(
            "max_forecast_horizon", 3
        )
        metadata["requires_exogenous_future"] = True

    logger.info(
        "champion_monthly_model built — family=%s  candidate=%s  refit=%s (n_obs=%s)",
        production_family,
        production_candidate_id,
        refit_info["performed"],
        refit_info["n_obs"],
    )
    return champion_model, metadata


# ── Node 5 ────────────────────────────────────────────────────────────────────


def generate_monthly_family_champion_explanations(  # noqa: PLR0913, PLR0912, PLR0915
    monthly_family_champion_summary: pd.DataFrame,
    monthly_prophet_candidate_models: dict,
    monthly_sarimax_candidate_models: dict,
    monthly_prophet_full_train: pd.DataFrame,
    monthly_sarimax_full_train: pd.DataFrame,
    monthly_sarimax_training_metadata: dict,
    params_monthly: dict,
    monthly_catboost_candidate_models: dict | None = None,
    monthly_catboost_full_train: pd.DataFrame | None = None,
    monthly_catboost_split_metadata: dict | None = None,
) -> tuple[pd.DataFrame, Any, pd.DataFrame, dict]:
    """Generate driver-importance explanations for every monthly family champion.

    For each family champion (Prophet, SARIMAX, CatBoost), the winning configuration is
    refit on full history (the same Champion Protocol stage-5 refit used for the
    production champion, via :func:`_build_production_champion_model`) so the explanation
    reflects the production-representative model. Importance is then computed with the
    method appropriate to each family — SHAP (``TreeExplainer``) for CatBoost, centered
    component contributions for Prophet, and absolute exogenous coefficients for SARIMAX
    (see ``explainability.py``).

    Explainability is auxiliary: a failure for one family is logged and skipped rather
    than breaking model selection, so the node always returns materialisable artifacts.

    Args:
        monthly_family_champion_summary: Family champions table (``family``,
            ``family_champion_id``) from :func:`select_monthly_family_champions`.
        monthly_prophet_candidate_models: Dict ``candidate_id -> Prophet model``.
        monthly_sarimax_candidate_models: Dict ``trial_id -> SARIMAX candidate entry``.
        monthly_catboost_candidate_models: Dict ``candidate_id -> CatBoost candidate entry``.
        monthly_prophet_full_train: Prophet-format full history (``ds``, ``y``, regressors).
        monthly_sarimax_full_train: SARIMAX-format full history (date, target, exog).
        monthly_catboost_full_train: CatBoost-format full history (features + target).
        monthly_sarimax_training_metadata: SARIMAX metadata (``exogenous_columns``).
        monthly_catboost_split_metadata: CatBoost split metadata (feature/date columns).
        params_monthly: Contents of ``model_selection.monthly`` (reads ``explainability``).

    Returns:
        Four-element tuple:

        1. ``monthly_family_champion_importance`` — long-form importance table across
           families (:data:`explainability.IMPORTANCE_COLUMNS`).
        2. ``monthly_catboost_shap_explainer`` — the fitted ``shap.TreeExplainer`` for the
           CatBoost champion (``None`` when CatBoost is not a champion / SHAP failed).
        3. ``monthly_catboost_shap_values`` — per-observation SHAP values (date + one
           column per feature); empty DataFrame when unavailable.
        4. ``monthly_family_champion_explainability_metadata`` — per-family method,
           champion id, and provenance.
    """
    computed_at = datetime.now(tz=UTC).isoformat()
    explain_cfg = dict(params_monthly.get("explainability", {}) or {})
    enabled = bool(explain_cfg.get("enabled", True))

    empty_importance = pd.DataFrame(columns=IMPORTANCE_COLUMNS)
    empty_shap = pd.DataFrame()
    metadata: dict = {
        "computed_at": computed_at,
        "granularity": "monthly",
        "enabled": enabled,
        "top_n_features": int(explain_cfg.get("top_n_features", 15)),
        "families": {},
        "n_families_explained": 0,
    }

    summary_empty = (
        monthly_family_champion_summary is None or monthly_family_champion_summary.empty
    )
    if not enabled or summary_empty:
        logger.info(
            "Family-champion explainability skipped (enabled=%s, summary_empty=%s).",
            enabled,
            summary_empty,
        )
        # Empty-dict sentinel (falsy) instead of None: Kedro forbids saving None,
        # and consumers treat a falsy explainer as "no SHAP available".
        return empty_importance, {}, empty_shap, metadata

    include_components = bool(
        explain_cfg.get("prophet", {}).get("include_components", True)
    )

    per_family: dict[str, dict] = {}
    families_meta: dict[str, dict] = {}
    catboost_explainer: Any = {}
    catboost_shap_values: pd.DataFrame = empty_shap

    for _, row in monthly_family_champion_summary.iterrows():
        family = str(row["family"])
        champion_id = str(row["family_champion_id"])
        try:
            candidate_model = _resolve_champion_model(
                production_family=family,
                production_candidate_id=champion_id,
                prophet_candidate_models=monthly_prophet_candidate_models,
                sarimax_candidate_models=monthly_sarimax_candidate_models,
                catboost_candidate_models=monthly_catboost_candidate_models or {},
            )
            champion_model, contract, _refit_info = _build_production_champion_model(
                production_family=family,
                candidate_model=candidate_model,
                prophet_full_train=monthly_prophet_full_train,
                sarimax_full_train=monthly_sarimax_full_train,
                sarimax_training_metadata=monthly_sarimax_training_metadata,
                params_monthly=params_monthly,
                catboost_full_train=monthly_catboost_full_train,
                catboost_split_metadata=monthly_catboost_split_metadata or {},
            )

            if family == "catboost":
                feature_columns = list(champion_model.get("feature_columns", []))
                model_obj = champion_model.get("model")
                missing = [
                    c
                    for c in feature_columns
                    if c not in monthly_catboost_full_train.columns
                ]
                if model_obj is None or not feature_columns or missing:
                    raise ValueError(
                        f"CatBoost SHAP unavailable (model={model_obj is not None}, "
                        f"n_features={len(feature_columns)}, missing={missing})."
                    )
                x_df = (
                    monthly_catboost_full_train[feature_columns]
                    .astype(float)
                    .reset_index(drop=True)
                )
                importance_df, shap_values_df, catboost_explainer, base_value = (
                    compute_catboost_shap_importance(model_obj, x_df, feature_columns)
                )
                date_col = str(
                    monthly_catboost_split_metadata.get(
                        "date_column", "month_start_date"
                    )
                )
                if date_col in monthly_catboost_full_train.columns:
                    shap_values_df.insert(
                        0,
                        date_col,
                        pd.to_datetime(
                            monthly_catboost_full_train[date_col]
                        ).reset_index(drop=True),
                    )
                catboost_shap_values = shap_values_df
                per_family[family] = {
                    "champion_id": champion_id,
                    "importance": importance_df,
                }
                families_meta[family] = {
                    "champion_id": champion_id,
                    "method": "shap_tree_explainer",
                    "base_value": base_value,
                    "n_obs": int(len(x_df)),
                    "n_features": len(feature_columns),
                    "feature_columns": feature_columns,
                }

            elif family == "prophet":
                active_regressors = list(contract.get("active_regressors", []))
                importance_df = compute_prophet_regressor_importance(
                    champion_model,
                    monthly_prophet_full_train,
                    active_regressors,
                    include_components=include_components,
                )
                per_family[family] = {
                    "champion_id": champion_id,
                    "importance": importance_df,
                }
                families_meta[family] = {
                    "champion_id": champion_id,
                    "method": "prophet_component_contribution",
                    "n_regressors": len(active_regressors),
                    "include_components": include_components,
                }

            elif family == "sarimax":
                results = (
                    champion_model.get("model")
                    if isinstance(champion_model, dict)
                    else None
                )
                used_exog = list(contract.get("active_regressors", []))
                if results is None:
                    raise ValueError("SARIMAX results object unavailable.")
                importance_df = compute_sarimax_coefficient_importance(
                    results, used_exog
                )
                per_family[family] = {
                    "champion_id": champion_id,
                    "importance": importance_df,
                }
                families_meta[family] = {
                    "champion_id": champion_id,
                    "method": "sarimax_coefficients",
                    "n_exog": len(used_exog),
                    "has_exog": bool(used_exog),
                }
            else:
                logger.warning(
                    "Unknown family '%s' in explainability; skipping.", family
                )
        except Exception as exc:  # explainability is auxiliary — never break selection
            logger.warning(
                "Explainability failed for family '%s' (champion=%s): %s",
                family,
                champion_id,
                exc,
            )
            families_meta[family] = {
                "champion_id": champion_id,
                "method": None,
                "error": str(exc),
            }

    importance_table = assemble_family_importance_table(per_family, computed_at)
    metadata["families"] = families_meta
    metadata["n_families_explained"] = len(per_family)
    logger.info(
        "Family-champion explainability built for %d/%d families: %s",
        len(per_family),
        len(monthly_family_champion_summary),
        list(per_family.keys()),
    )
    return importance_table, catboost_explainer, catboost_shap_values, metadata


# ── Private helpers ───────────────────────────────────────────────────────────


def _build_sort_columns(
    primary_metric: str, tie_breakers: list[str], available_columns: Any
) -> list[str]:
    """Return an ordered list of column names to sort by, skipping unavailable ones."""
    cols: list[str] = []
    for name in [primary_metric, *tie_breakers]:
        col = "abs_bias" if name == "abs_bias" else name
        if col in available_columns:
            cols.append(col)
    return cols or [primary_metric]


def _resolve_champion_model(
    production_family: str,
    production_candidate_id: str,
    prophet_candidate_models: dict,
    sarimax_candidate_models: dict,
    catboost_candidate_models: dict,
) -> Any:
    """Return the model object for the elected production champion."""
    if production_family == "prophet":
        if production_candidate_id not in prophet_candidate_models:
            raise RuntimeError(
                f"Production champion Prophet candidate '{production_candidate_id}' "
                "not found in monthly_prophet_candidate_models."
            )
        return prophet_candidate_models[production_candidate_id]
    if production_family == "sarimax":
        if production_candidate_id not in sarimax_candidate_models:
            raise RuntimeError(
                f"Production champion SARIMAX candidate '{production_candidate_id}' "
                "not found in monthly_sarimax_candidate_models."
            )
        # The SARIMAX entry is a dict; return it whole so the champion model
        # preserves the config and validation_metrics alongside the fitted result.
        return sarimax_candidate_models[production_candidate_id]
    if production_family == "catboost":
        if production_candidate_id not in catboost_candidate_models:
            raise RuntimeError(
                f"Production champion CatBoost candidate '{production_candidate_id}' "
                "not found in monthly_catboost_candidate_models."
            )
        # The CatBoost entry is a dict carrying the fitted model, config, and
        # feature_columns; return it whole so the inference contract is preserved.
        return catboost_candidate_models[production_candidate_id]
    raise ValueError(
        f"Unknown production champion family '{production_family}'. "
        "Expected 'prophet', 'sarimax', or 'catboost'."
    )


def _build_production_champion_model(  # noqa: PLR0913
    production_family: str,
    candidate_model: Any,
    prophet_full_train: pd.DataFrame,
    sarimax_full_train: pd.DataFrame,
    sarimax_training_metadata: dict,
    params_monthly: dict,
    catboost_full_train: pd.DataFrame | None = None,
    catboost_split_metadata: dict | None = None,
) -> tuple[Any, dict, dict]:
    """Refit the elected champion on full history and build its inference contract.

    Implements Champion Protocol stage 5. The winning configuration is refit on all
    available history so production forecasts use the full training window. When
    ``refit_champion.enabled`` is false, the train-only candidate is returned
    unchanged (useful for fast debugging runs).

    Args:
        production_family: Elected production champion family.
        candidate_model: Train-only champion artifact resolved from the candidate
            pool (Prophet model, SARIMAX candidate dict, or CatBoost candidate dict).
        prophet_full_train: Prophet-format full history (ds, y, regressors).
        sarimax_full_train: SARIMAX-format full history (date, target, exog).
        sarimax_training_metadata: SARIMAX training metadata (exogenous_columns).
        params_monthly: Contents of ``model_selection.monthly``.
        catboost_full_train: CatBoost-format full history (precomputed features +
            target). Required only for a CatBoost champion.
        catboost_split_metadata: CatBoost split metadata (feature/categorical/target/
            date columns). Required only for a CatBoost champion.

    Returns:
        Tuple ``(champion_model, inference_contract, refit_info)``.
    """
    refit_enabled = bool(params_monthly.get("refit_champion", {}).get("enabled", True))

    if production_family == "prophet":
        active_regressors = _prophet_regressor_names(candidate_model)
        if refit_enabled:
            champion_model = _refit_prophet_on_frame(
                candidate_model, prophet_full_train, active_regressors
            )
            refit_info = _refit_info(True, prophet_full_train, "ds")
        else:
            champion_model = candidate_model
            refit_info = _refit_info(False, prophet_full_train, "ds")
        contract = _build_inference_contract("prophet", active_regressors)
        return champion_model, contract, refit_info

    if production_family == "sarimax":
        config = (
            dict(candidate_model.get("config", {}))
            if isinstance(candidate_model, dict)
            else {}
        )
        exog_cols = list(sarimax_training_metadata.get("exogenous_columns", []))
        date_col = "month_start_date"
        if refit_enabled:
            champion_model, used_exog = _refit_sarimax_full_history(
                config, sarimax_full_train, exog_cols
            )
            refit_info = _refit_info(True, sarimax_full_train, date_col)
        else:
            champion_model = candidate_model
            used_exog = exog_cols if config.get("use_exog", False) else []
            refit_info = _refit_info(False, sarimax_full_train, date_col)
        contract = _build_inference_contract(
            "sarimax", used_exog, sarimax_config=config
        )
        return champion_model, contract, refit_info

    if production_family == "catboost":
        return _build_catboost_champion_model(
            candidate_model=candidate_model,
            catboost_full_train=catboost_full_train,
            catboost_split_metadata=catboost_split_metadata or {},
            refit_enabled=refit_enabled,
        )

    raise ValueError(
        f"Cannot build a production champion model for unknown family "
        f"'{production_family}'."
    )


def _build_catboost_champion_model(
    candidate_model: Any,
    catboost_full_train: pd.DataFrame | None,
    catboost_split_metadata: dict,
    refit_enabled: bool,
) -> tuple[Any, dict, dict]:
    """Refit a direct multi-horizon CatBoost champion on full history.

    The CatBoost champion artifact from the direct multi-horizon training node carries three
    fitted models (``model_h1``, ``model_h2``, ``model_h3``) and shared
    ``feature_columns``. On refit, three fresh models are trained with the same
    config on full history. The champion artifact preserves the three-model structure
    so the inference adapter can apply each model independently (no recursion).

    Args:
        candidate_model: CatBoost candidate entry dict carrying ``config``,
            ``feature_columns``, and optionally ``model_h1/h2/h3``.
        catboost_full_train: CatBoost-format full history (features + target).
        catboost_split_metadata: CatBoost split metadata (column names, direct_multi_horizon).
        refit_enabled: Whether to refit on full history (Champion Protocol stage 5).

    Returns:
        Tuple ``(champion_model, inference_contract, refit_info)``.
    """
    if not isinstance(candidate_model, dict):
        raise ValueError(
            "CatBoost champion artifact must be a candidate entry dict; "
            f"received {type(candidate_model)!r}."
        )

    config = dict(candidate_model.get("config", {}))
    feature_columns = list(
        candidate_model.get("feature_columns")
        or catboost_split_metadata.get("all_feature_columns", [])
    )
    categorical_features = list(
        catboost_split_metadata.get("categorical_feature_columns", [])
    )
    target_col = str(catboost_split_metadata.get("target_column", "monthly_demand"))
    date_col = str(catboost_split_metadata.get("date_column", "month_start_date"))
    sku_col = str(catboost_split_metadata.get("sku_column", "sku"))
    horizons = list(
        catboost_split_metadata.get("direct_multi_horizon", {}).get(
            "horizons", [1, 2, 3]
        )
    )

    if not feature_columns:
        raise ValueError("Cannot resolve CatBoost feature_columns for the champion.")

    if refit_enabled:
        if catboost_full_train is None or catboost_full_train.empty:
            raise ValueError(
                "CatBoost champion refit requested but monthly_catboost_full_train is "
                "empty or was not provided to monthly model selection."
            )
        champion_model = _refit_catboost_direct_full_history(
            config=config,
            full_train_df=catboost_full_train,
            feature_columns=feature_columns,
            target_col=target_col,
            date_col=date_col,
            sku_col=sku_col,
            horizons=horizons,
        )
        refit_info = _refit_info(True, catboost_full_train, date_col)
    else:
        champion_model = candidate_model
        refit_info = _refit_info(
            False,
            catboost_full_train if catboost_full_train is not None else pd.DataFrame(),
            date_col,
        )

    contract = _build_inference_contract(
        "catboost",
        feature_columns,
        catboost_feature_columns=feature_columns,
        catboost_categorical_features=categorical_features,
        catboost_target_column=target_col,
        catboost_date_column=date_col,
        catboost_strategy="direct_multi_horizon",
        catboost_max_horizon=max(horizons) if horizons else 3,
    )
    return champion_model, contract, refit_info


def _refit_catboost_direct_full_history(
    config: dict,
    full_train_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    date_col: str,
    sku_col: str,
    horizons: list[int],
) -> dict:
    """Refit a direct multi-horizon CatBoost champion on full history.

    Trains one CatBoostRegressor per horizon on (features_at_t, demand(t+h)) pairs.
    Returns a champion artifact dict with keys ``model_h1``, ``model_h2``,
    ``model_h3`` plus ``feature_columns`` and ``refit_scope``.
    """
    from catboost import CatBoostRegressor

    df = full_train_df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    available = [c for c in feature_columns if c in df.columns]
    missing = [c for c in feature_columns if c not in available]
    if missing:
        logger.warning(
            "CatBoost full-history direct refit: feature columns %s not found — proceeding with %d.",
            missing,
            len(available),
        )
    if not available:
        raise ValueError(
            "No CatBoost feature columns present in full_train_df; cannot refit champion."
        )

    fit_cfg = {k: v for k, v in config.items() if k != "random_seed"}
    fit_cfg["random_seed"] = config.get("random_seed", 42)

    champion: dict = {
        "model_family": "catboost",
        "granularity": "monthly",
        "strategy": "direct_multi_horizon",
        "config": config,
        "feature_columns": available,
        "refit_scope": "full_history",
    }
    for h in horizons:
        y_h = df.groupby(sku_col, sort=False)[target_col].transform(
            lambda s, shift=h: s.shift(-shift)
        )
        valid_mask = y_h.notna()
        X_full = df.loc[valid_mask, available].to_numpy(dtype=float)
        y_full = y_h[valid_mask].to_numpy(dtype=float)

        model_h = CatBoostRegressor(**fit_cfg, verbose=False, allow_writing_files=False)
        model_h.fit(X_full, y_full, verbose=False)
        champion[f"model_h{h}"] = model_h
        logger.info(
            "CatBoost direct full-history refit: model_h%d trained on %d pairs.",
            h,
            len(y_full),
        )

    return champion


def _prophet_regressor_names(model: Any) -> list[str]:
    """Return the regressor names registered on a fitted Prophet model."""
    extra = getattr(model, "extra_regressors", None)
    return list(extra.keys()) if isinstance(extra, dict) else []


def _refit_prophet_on_frame(
    candidate_model: Any,
    fit_df: pd.DataFrame,
    active_regressors: list[str],
    date_col: str = "ds",
    target_col: str = "y",
) -> Any:
    """Rebuild a Prophet model from a candidate's configuration and fit it on ``fit_df``.

    Hyperparameters and regressor modes are read from the fitted candidate so the
    refit reproduces the selected configuration exactly, on a different data window
    (train+validation for test scoring, or full history for the champion). Only
    regressors present in ``fit_df`` are used, so minimal frames degrade gracefully.

    Returns:
        The newly fitted Prophet model.
    """
    model_params = _extract_prophet_params(candidate_model)
    extra = getattr(candidate_model, "extra_regressors", None) or {}
    regressors = [r for r in active_regressors if r in fit_df.columns]

    prepared = fit_df.copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col])
    prepared = prepared.sort_values(date_col).reset_index(drop=True)
    fit_input = prepared[[date_col, target_col, *regressors]].rename(
        columns={date_col: "ds", target_col: "y"}
    )

    model = Prophet(**model_params)
    for name in regressors:
        reg_cfg = extra.get(name) if isinstance(extra.get(name), dict) else {}
        kwargs: dict = {"mode": reg_cfg.get("mode", "additive")}
        # Reproduce the candidate's tuned per-regressor prior scale. Without this,
        # add_regressor() silently defaults each regressor to holidays_prior_scale,
        # discarding the prior scales chosen during tuning.
        prior_scale = reg_cfg.get("prior_scale")
        if prior_scale is not None:
            kwargs["prior_scale"] = float(prior_scale)
        model.add_regressor(name, **kwargs)
    model.fit(fit_input)
    return model


def _extract_prophet_params(model: Any) -> dict:
    """Extract Prophet constructor hyperparameters from a fitted model (with defaults)."""
    return {
        "changepoint_prior_scale": float(
            getattr(model, "changepoint_prior_scale", 0.05)
        ),
        "seasonality_prior_scale": float(
            getattr(model, "seasonality_prior_scale", 10.0)
        ),
        "holidays_prior_scale": float(getattr(model, "holidays_prior_scale", 10.0)),
        "seasonality_mode": str(getattr(model, "seasonality_mode", "additive")),
        "yearly_seasonality": getattr(model, "yearly_seasonality", True),
        "weekly_seasonality": getattr(model, "weekly_seasonality", False),
        "daily_seasonality": getattr(model, "daily_seasonality", False),
        "interval_width": float(getattr(model, "interval_width", 0.8)),
    }


def _extract_prophet_regressor_prior_scales(model: Any) -> dict:
    """Extract the per-regressor prior scales from a fitted Prophet model.

    Prophet stores each registered regressor's prior scale under
    ``model.extra_regressors[name]['prior_scale']``. These are the tuned
    regularisation strengths per exogenous driver (e.g. the demand-planning
    regressors), which the global hyperparameters do not capture. Regressors that
    were not individually tuned carry the global ``holidays_prior_scale`` default.

    Args:
        model: A fitted Prophet model.

    Returns:
        Mapping of regressor name to prior scale, ordered as registered. Empty
        when the model exposes no extra regressors.
    """
    extra = getattr(model, "extra_regressors", None) or {}
    prior_scales: dict = {}
    for name, cfg in extra.items():
        if isinstance(cfg, dict) and cfg.get("prior_scale") is not None:
            prior_scales[str(name)] = float(cfg["prior_scale"])
    return prior_scales


def _refit_sarimax_full_history(
    config: dict,
    full_train_df: pd.DataFrame,
    exog_cols: list[str],
    date_col: str = "month_start_date",
    target_col: str = "monthly_demand",
) -> tuple[dict, list[str]]:
    """Refit a SARIMAX champion on full history using the candidate's order config.

    Returns the refit in the candidate-dict shape so the inference adapter unwraps
    it uniformly, plus the exogenous column names actually used.

    Returns:
        Tuple ``(champion_entry, used_exog_cols)``.
    """
    df = full_train_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    y = df[target_col].to_numpy(dtype=float)
    available_exog = [c for c in exog_cols if c in df.columns]
    missing_exog = [c for c in exog_cols if c not in df.columns]
    if missing_exog:
        logger.warning(
            "SARIMAX full-history refit: exogenous columns %s not found in "
            "full_train_df. Proceeding with available columns: %s",
            missing_exog,
            available_exog,
        )
    use_exog = bool(config.get("use_exog", False)) and bool(available_exog)
    exog = df[available_exog].to_numpy(dtype=float) if use_exog else None

    model = SARIMAX(
        endog=y,
        exog=exog,
        order=tuple(config["order"]),
        seasonal_order=tuple(config["seasonal_order"]),
        trend=config.get("trend"),
        enforce_stationarity=bool(config.get("enforce_stationarity", False)),
        enforce_invertibility=bool(config.get("enforce_invertibility", False)),
    )
    result = model.fit(disp=False)

    champion_entry = {
        "model_family": "sarimax",
        "granularity": "monthly",
        "config": config,
        "model": result,
        "refit_scope": "full_history",
    }
    return champion_entry, (available_exog if use_exog else [])


def _refit_info(performed: bool, full_train_df: pd.DataFrame, date_col: str) -> dict:
    """Build the refit audit block for champion metadata."""
    dates = (
        pd.to_datetime(full_train_df[date_col])
        if date_col in full_train_df.columns
        else None
    )
    return {
        "performed": performed,
        "data_scope": "full_history",  # train + validation + test
        "n_obs": int(len(full_train_df)),
        "start_date": (
            str(dates.min().date()) if dates is not None and not dates.empty else None
        ),
        "end_date": (
            str(dates.max().date()) if dates is not None and not dates.empty else None
        ),
        "refit_at": datetime.now(tz=UTC).isoformat(),
        "note": (
            "Champion refit on all available history for production forecasting. "
            "Reported test metrics come from the pre-refit selection stage."
        ),
    }


def _extract_champion_hyperparameters(
    production_family: str,
    champion_model: Any,
    inference_contract: dict,
) -> dict:
    """Extract the champion's key hyperparameters for audit and downstream inspection.

    Args:
        production_family: Elected production champion family.
        champion_model: The fitted champion model (Prophet) or candidate dict (SARIMAX).
        inference_contract: Inference contract dict containing SARIMAX config when applicable.

    Returns:
        Dict of hyperparameter names to values; schema is family-specific.
    """
    if production_family == "prophet":
        params = _extract_prophet_params(champion_model)
        params["regressor_prior_scales"] = _extract_prophet_regressor_prior_scales(
            champion_model
        )
        return params
    if production_family == "sarimax":
        config = dict(inference_contract.get("sarimax_config") or {})
        return {
            "order": config.get("order"),
            "seasonal_order": config.get("seasonal_order"),
            "trend": config.get("trend"),
            "use_exog": bool(config.get("use_exog", False)),
        }
    if production_family == "catboost":
        config = (
            dict(champion_model.get("config", {}))
            if isinstance(champion_model, dict)
            else {}
        )
        return {str(k): _to_native(v) for k, v in config.items()}
    return {}


def _build_inference_contract(  # noqa: PLR0913
    production_family: str,
    active_regressors: list[str],
    sarimax_config: dict | None = None,
    catboost_feature_columns: list[str] | None = None,
    catboost_categorical_features: list[str] | None = None,
    catboost_target_column: str | None = None,
    catboost_date_column: str | None = None,
    catboost_strategy: str | None = None,
    catboost_max_horizon: int = 3,
) -> dict:
    """Describe how the champion must be consumed at inference time."""
    if production_family == "prophet":
        return {
            "model_family": "prophet",
            "date_column": "ds",
            "target_column": "y",
            "sku_column": "sku",
            "active_regressors": list(active_regressors),
            "forecast_horizons": [3, 6, 12],
            "has_prediction_intervals": True,
            "interval_method": "prophet_native",
            "sarimax_config": None,
        }

    if production_family == "sarimax":
        config = dict(sarimax_config or {})
        return {
            "model_family": "sarimax",
            "date_column": "month_start_date",
            "target_column": "monthly_demand",
            "sku_column": "sku",
            "active_regressors": list(active_regressors),
            "forecast_horizons": [3, 6, 12],
            "has_prediction_intervals": True,
            "interval_method": "sarimax_get_forecast_conf_int",
            "sarimax_config": {
                "order": config.get("order"),
                "seasonal_order": config.get("seasonal_order"),
                "trend": config.get("trend"),
                "use_exog": bool(config.get("use_exog", False)),
                "enforce_stationarity": bool(config.get("enforce_stationarity", False)),
                "enforce_invertibility": bool(
                    config.get("enforce_invertibility", False)
                ),
            },
        }

    if production_family == "catboost":
        feature_columns = list(catboost_feature_columns or active_regressors)
        strategy = catboost_strategy or "direct_multi_horizon"
        # CatBoost direct multi-horizon: max 3 months.
        forecast_horizons = list(range(1, catboost_max_horizon + 1))
        return {
            "model_family": "catboost",
            "date_column": catboost_date_column or "month_start_date",
            "target_column": catboost_target_column or "monthly_demand",
            "sku_column": "sku",
            "active_regressors": list(feature_columns),
            "feature_columns": list(feature_columns),
            "categorical_features": list(catboost_categorical_features or []),
            "forecast_horizons": forecast_horizons,
            "max_forecast_horizon": catboost_max_horizon,
            "has_prediction_intervals": False,
            "interval_method": None,
            "strategy": strategy,
            "sarimax_config": None,
        }

    raise ValueError(
        f"Cannot build an inference contract for unknown family '{production_family}'."
    )


def _extract_champion_metrics(
    metrics_df: pd.DataFrame, family: str, candidate_id: str
) -> dict:
    """Extract the rolling-origin metric set for the production champion."""
    keys = ["wmape", "wmape_m1", "wmape_m2", "wmape_m3", "mase", "bias", "rmse"]
    mask = (metrics_df["family"] == family) & (
        metrics_df["candidate_id"] == candidate_id
    )
    subset = metrics_df[mask]
    if subset.empty:
        return dict.fromkeys(keys)
    row = subset.iloc[0]
    return {k: _safe_float(row.get(k)) for k in keys}


def _log_production_champion_comparison(
    ranked: pd.DataFrame,
    primary_metric: str,
    tie_breakers: list[str],
) -> None:
    """Log a ranked side-by-side comparison of all family champions.

    Shows all rolling-origin metrics for every family champion, ranked by the
    primary metric, so it is unambiguous why the production champion was elected.
    """
    best_val = _safe_float(ranked.iloc[0].get(primary_metric)) if not ranked.empty else None
    tb_str = ", ".join(str(t) for t in tie_breakers)
    lines = [
        f"Monthly production champion selection — {len(ranked)} families "
        f"ranked by {primary_metric}  (tie-breakers: {tb_str}):"
    ]
    for pos, (_, row) in enumerate(ranked.iterrows(), start=1):
        family = str(row.get("family", "?"))
        cid = str(row.get("family_champion_id", "?"))
        wmape = _safe_float(row.get("wmape"))
        wmape_m1 = _safe_float(row.get("wmape_m1"))
        wmape_m2 = _safe_float(row.get("wmape_m2"))
        wmape_m3 = _safe_float(row.get("wmape_m3"))
        mase = _safe_float(row.get("mase"))
        bias = _safe_float(row.get("bias"))
        rmse = _safe_float(row.get("rmse"))

        wmape_s = f"{wmape:.4f}" if wmape is not None else "nan"
        wmape_m1_s = f"{wmape_m1:.4f}" if wmape_m1 is not None else "nan"
        wmape_m2_s = f"{wmape_m2:.4f}" if wmape_m2 is not None else "nan"
        wmape_m3_s = f"{wmape_m3:.4f}" if wmape_m3 is not None else "nan"
        mase_s = f"{mase:.4f}" if mase is not None else "n/a"
        bias_s = f"{bias:+.4f}" if bias is not None else "n/a"
        rmse_s = f"{rmse:.2f}" if rmse is not None else "n/a"

        if pos == 1:
            suffix = "  ★ PRODUCTION CHAMPION"
        else:
            cur_val = _safe_float(row.get(primary_metric))
            if cur_val is not None and best_val is not None:
                suffix = f"  (Δ{primary_metric}={cur_val - best_val:+.4f})"
            else:
                suffix = ""

        lines.append(
            f"  #{pos} {family:<10} {cid}"
            f"  wmape={wmape_s}  wmape_m1={wmape_m1_s}  wmape_m2={wmape_m2_s}"
            f"  wmape_m3={wmape_m3_s}  mase={mase_s}  bias={bias_s}  rmse={rmse_s}"
            f"{suffix}"
        )
    logger.info("\n".join(lines))


def _safe_float(value: Any) -> float | None:
    """Convert to Python float, returning None for missing or non-finite values."""
    if value is None:
        return None
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert to Python int, returning None for missing or non-finite values."""
    if value is None:
        return None
    try:
        f = float(value)
        return int(f) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _to_native(value: Any) -> Any:
    """Convert numpy scalar types to native Python types for JSON serialisation.

    CatBoost hyperparameter values read back from a candidate config may be numpy
    scalars (pandas coerces numeric columns). ``JSONDataset`` cannot serialise numpy
    scalars, so they are converted here. Non-numpy values are returned unchanged.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value
