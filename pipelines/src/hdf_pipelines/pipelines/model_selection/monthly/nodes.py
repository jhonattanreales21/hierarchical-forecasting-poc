"""Monthly multi-family model selection nodes.

Compares prechampion candidates on the held-out monthly test
period, selects one family champion per family, then elects one monthly production
champion.
"""

import logging
import warnings as _warnings
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from prophet import Prophet
from shared.catboost_recursive import (
    build_demand_buffer,
    compute_recursive_target_features,
    extract_periods_from_column_names,
)
from shared.metrics import mase as _shared_mase
from shared.metrics import rmse as _shared_rmse
from shared.metrics import wape as _shared_wape
from statsmodels.stats.diagnostic import acorr_ljungbox
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


def evaluate_monthly_family_candidates_on_test(  # noqa: PLR0913
    monthly_prophet_candidate_models: dict,
    monthly_prophet_prechampion_configs: dict,
    monthly_prophet_train: pd.DataFrame,
    monthly_prophet_validation: pd.DataFrame,
    monthly_prophet_test: pd.DataFrame,
    monthly_sarimax_candidate_models: dict,
    monthly_sarimax_prechampion_configs: dict,
    monthly_sarimax_training_metadata: dict,
    monthly_sarimax_train: pd.DataFrame,
    monthly_sarimax_validation: pd.DataFrame,
    monthly_sarimax_test: pd.DataFrame,
    params_monthly: dict,
    params_prophet: dict,
    params_sarimax: dict,
    monthly_catboost_candidate_models: dict | None = None,
    monthly_catboost_prechampion_configs: dict | None = None,
    monthly_catboost_split_metadata: dict | None = None,
    monthly_catboost_train: pd.DataFrame | None = None,
    monthly_catboost_validation: pd.DataFrame | None = None,
    monthly_catboost_test: pd.DataFrame | None = None,
    params_catboost: dict | None = None,
) -> pd.DataFrame:
    """Score all Prophet, SARIMAX, and CatBoost prechampions on the held-out test set.

    Evaluates only the families listed in ``params_monthly.active_families``. Every
    family is scored consistently: each candidate is refit on the combined
    train+validation split and then forecast forward over the held-out test period.
    The test split is never seen during fitting, so the reported test metrics are
    leakage-safe and the cross-family comparison is fair (all families see the same
    data at test time). A full-history refit (including the test split) happens only
    later, for the elected production champion.

    CatBoost inputs are optional: they are only required when ``catboost`` is an
    active family. When present, CatBoost candidates are refit on train+validation
    and scored on the precomputed test feature matrix (a one-shot tabular prediction,
    consistent with how CatBoost validation metrics were computed during training).

    Args:
        monthly_prophet_candidate_models: Dict mapping candidate_id → fitted
            Prophet model from the training stage.
        monthly_prophet_prechampion_configs: Prechampion config dict emitted by
            Prophet training, containing a ``prechampions`` list.
        monthly_prophet_train: Training-only split; the leakage-safe MASE
            denominator and the first part of the Prophet refit window.
        monthly_prophet_validation: Validation split appended to train to form the
            leakage-safe Prophet refit window for test scoring.
        monthly_prophet_test: Held-out Prophet test split with ds, y, and regressor
            columns.
        monthly_sarimax_candidate_models: Dict mapping trial_id → SARIMAX candidate
            entry (rank, config, model, validation_metrics).
        monthly_sarimax_prechampion_configs: Prechampion config dict emitted by
            SARIMAX training, containing a ``candidates`` list.
        monthly_sarimax_training_metadata: Training metadata dict; used to retrieve
            the exogenous column names used during training.
        monthly_sarimax_train: Training-only split used as the leakage-safe MASE
            denominator and as the first part of the SARIMAX fit window.
        monthly_sarimax_validation: Validation split appended to train to form the
            leakage-safe SARIMAX fit window for test scoring.
        monthly_sarimax_test: Held-out SARIMAX test split with date, target, and
            optional exogenous columns.
        params_monthly: Contents of ``model_selection.monthly`` from the parameter
            file (active_families, primary_metric, mase_seasonal_period, …).
        params_prophet: Contents of ``model_selection.monthly_prophet`` (column
            names, metric settings, …).
        params_sarimax: Contents of ``model_selection.monthly_sarimax`` (column
            names, …).
        monthly_catboost_candidate_models: Dict mapping candidate_id → CatBoost
            candidate entry (rank, config, model, feature_columns, validation_metrics).
            Required only when ``catboost`` is an active family.
        monthly_catboost_prechampion_configs: Prechampion config dict emitted by
            CatBoost training, containing a ``candidates`` list. Required only when
            ``catboost`` is an active family.
        monthly_catboost_split_metadata: CatBoost split metadata (feature/target/date
            column names). Used to resolve feature columns at scoring time.
        monthly_catboost_train: CatBoost training-only split (MASE denominator and
            first part of the refit window).
        monthly_catboost_validation: CatBoost validation split appended to train to
            form the leakage-safe refit window for test scoring.
        monthly_catboost_test: Held-out CatBoost test split with precomputed feature
            columns and the target column.
        params_catboost: Contents of ``model_selection.monthly_catboost`` (column
            names). Optional; defaults are read from the split metadata.

    Returns:
        DataFrame with one row per evaluated candidate containing: family,
        candidate_id, candidate_rank, granularity, selection_stage,
        test_start_date, test_end_date, n_test_rows, wape, mase, rmse, bias,
        primary_metric, primary_metric_value, is_family_champion,
        is_production_champion.

    Raises:
        ValueError: When an active family has no prechampion candidates.
        RuntimeError: When all candidates for an active family fail to score.
    """
    active_families: list[str] = list(
        params_monthly.get("active_families", ["prophet", "sarimax"])
    )
    require_all: bool = bool(params_monthly.get("require_all_active_families", True))
    mase_period: int = int(params_monthly.get("mase_seasonal_period", 12))

    _validate_active_families(
        active_families,
        monthly_prophet_prechampion_configs,
        monthly_sarimax_prechampion_configs,
        monthly_catboost_prechampion_configs or {},
        require_all,
    )

    all_rows: list[dict] = []

    if "prophet" in active_families:
        prophet_rows = _score_prophet_candidates(
            candidate_models=monthly_prophet_candidate_models,
            prechampion_configs=monthly_prophet_prechampion_configs,
            train_df=monthly_prophet_train,
            validation_df=monthly_prophet_validation,
            test_df=monthly_prophet_test,
            params=params_prophet,
            mase_period=mase_period,
            require_all=require_all,
        )
        all_rows.extend(prophet_rows)

    if "sarimax" in active_families:
        sarimax_rows = _score_sarimax_candidates(
            candidate_models=monthly_sarimax_candidate_models,
            prechampion_configs=monthly_sarimax_prechampion_configs,
            training_metadata=monthly_sarimax_training_metadata,
            train_df=monthly_sarimax_train,
            validation_df=monthly_sarimax_validation,
            test_df=monthly_sarimax_test,
            params=params_sarimax,
            mase_period=mase_period,
            require_all=require_all,
        )
        all_rows.extend(sarimax_rows)

    if "catboost" in active_families:
        catboost_rows = _score_catboost_candidates(
            candidate_models=monthly_catboost_candidate_models or {},
            prechampion_configs=monthly_catboost_prechampion_configs or {},
            split_metadata=monthly_catboost_split_metadata or {},
            train_df=monthly_catboost_train,
            validation_df=monthly_catboost_validation,
            test_df=monthly_catboost_test,
            params=params_catboost or {},
            mase_period=mase_period,
            require_all=require_all,
        )
        all_rows.extend(catboost_rows)

    if not all_rows:
        raise RuntimeError("No candidate metrics were produced for any active family.")

    metrics_df = pd.DataFrame(all_rows)
    logger.info(
        "Monthly candidate test metrics — %d rows | families=%s",
        len(metrics_df),
        sorted(metrics_df["family"].unique().tolist()),
    )
    return metrics_df


# ── Node 2 ────────────────────────────────────────────────────────────────────


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
    """Assemble the cross-family candidate metrics table from rolling-origin pre-champions.

    Replaces the held-out test stage (protocol §4, §11): champions are selected
    directly from the pooled rolling-origin metrics already produced by each
    family tuner. One row per pre-champion candidate, with the WMAPE / WMAPE_Mh /
    MASE / BIAS metric set plus the SARIMAX Ljung-Box eligibility flag.

    Args:
        monthly_prophet_prechampion_configs: Prophet ``prechampion_configs`` artifact.
        monthly_sarimax_prechampion_configs: SARIMAX ``prechampion_configs`` artifact.
        params_monthly: Contents of ``model_selection.monthly`` (reads ``active_families``).
        monthly_catboost_prechampion_configs: CatBoost ``prechampion_configs`` artifact
            (optional; required only when ``catboost`` is in ``active_families``).

    Returns:
        DataFrame with one row per candidate across active families.

    Raises:
        ValueError: When no candidates can be assembled.
    """
    active_families = [
        str(f) for f in params_monthly.get("active_families", ["prophet", "sarimax"])
    ]
    rows: list[dict] = []
    if "prophet" in active_families:
        rows.extend(
            _candidate_rows(
                monthly_prophet_prechampion_configs, "prophet", "prechampions"
            )
        )
    if "sarimax" in active_families:
        rows.extend(
            _candidate_rows(
                monthly_sarimax_prechampion_configs, "sarimax", "candidates"
            )
        )
    if "catboost" in active_families:
        rows.extend(
            _candidate_rows(
                monthly_catboost_prechampion_configs or {}, "catboost", "candidates"
            )
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
    applies tie-breakers in order (protocol §3.10). Exactly one family champion is
    selected per active family. For SARIMAX, candidates flagged by the Ljung-Box
    filter are excluded before ranking (protocol §3.9), with a fallback to the full
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

        # SARIMAX: exclude Ljung-Box-flagged candidates before ranking (protocol §3.9).
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
        logger.info(
            "Family champion (%s): %s  %s=%.4f",
            family,
            best["candidate_id"],
            primary_metric,
            _safe_float(best.get(primary_metric)) or float("nan"),
        )

    if not champion_rows:
        raise RuntimeError(
            "No family champions could be selected from the metrics table."
        )
    return pd.DataFrame(champion_rows)


# ── Node 3 ────────────────────────────────────────────────────────────────────


def select_monthly_production_champion(
    monthly_family_champion_summary: pd.DataFrame,
    monthly_candidate_metrics: pd.DataFrame,
    params_monthly: dict,
) -> pd.DataFrame:
    """Compare family champions and elect one monthly production champion.

    Only family champions (one per family) participate in the final comparison.
    The selection uses the same metric ordering as family champion selection:
    primary metric first, then tie-breakers in order (protocol §3.10).

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

    logger.info(
        "Monthly production champion: family=%s  candidate=%s  %s=%.4f",
        best["family"],
        best["family_champion_id"],
        primary_metric,
        _safe_float(best.get(primary_metric)) or float("nan"),
    )

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


# ── Node 4 ────────────────────────────────────────────────────────────────────


def annotate_monthly_candidate_champion_flags(
    monthly_candidate_metrics: pd.DataFrame,
    monthly_family_champion_summary: pd.DataFrame,
    monthly_model_selection_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Mark family and production champions in the candidate metrics table.

    ``monthly_candidate_test_metrics`` is the app/reporting-facing audit table, so
    its champion flags must reflect the separately persisted family and production
    champion summaries.

    Args:
        monthly_candidate_test_metrics: Full held-out test metrics for all
            candidates.
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


# ── Node 5 ────────────────────────────────────────────────────────────────────


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
        monthly_candidate_test_metrics: Full candidate metrics table.
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
                "no reserved out-of-sample window was used (optimistic-selection risk, "
                "protocol §9.1)."
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


# ── Node 6 ────────────────────────────────────────────────────────────────────


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


def _concat_sorted(frames: list[pd.DataFrame], date_col: str) -> pd.DataFrame:
    """Concatenate frames, parse the date column, sort ascending, and reset the index.

    Empty or None frames are ignored so a missing validation split degrades to the
    train split alone rather than failing.
    """
    parts = [f for f in frames if f is not None and not f.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if combined.empty:
        return combined
    combined[date_col] = pd.to_datetime(combined[date_col])
    return combined.sort_values(date_col).reset_index(drop=True)


def _validate_active_families(
    active_families: list[str],
    prophet_configs: dict,
    sarimax_configs: dict,
    catboost_configs: dict,
    require_all: bool,
) -> None:
    """Raise clearly when required family artifacts are absent."""
    missing: list[str] = []
    if "prophet" in active_families and not prophet_configs.get("prechampions"):
        missing.append(
            "prophet (monthly_prophet_prechampion_configs has no prechampions)"
        )
    if "sarimax" in active_families and not sarimax_configs.get("candidates"):
        missing.append(
            "sarimax (monthly_sarimax_prechampion_configs has no candidates)"
        )
    if "catboost" in active_families and not catboost_configs.get("candidates"):
        missing.append(
            "catboost (monthly_catboost_prechampion_configs has no candidates)"
        )
    if missing and require_all:
        raise ValueError(
            "require_all_active_families=true but the following families have no "
            f"prechampion artifacts: {missing}"
        )
    for msg in missing:
        logger.warning("Active family missing artifacts: %s", msg)


def _score_prophet_candidates(  # noqa: PLR0913
    candidate_models: dict,
    prechampion_configs: dict,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    mase_period: int,
    require_all: bool,
) -> list[dict]:
    """Score each Prophet prechampion leakage-safely on the held-out test split.

    Each candidate is refit on train + validation only (mirroring the SARIMAX path),
    then used to forecast the test period. The MASE denominator uses the train-only
    split. The test split is never seen during fitting.

    When ``operational_lead_time`` is enabled, rolling-origin M-h WAPE is also
    computed (a fresh Prophet refit per origin, forecasting h steps ahead), giving the
    same operational lead-time diagnostics that SARIMAX already produces.
    """
    date_col: str = params.get("date_column", "ds")
    target_col: str = params.get("target_column", "y")

    # Rolling-origin (operational lead-time) config — disabled unless configured.
    ro_cfg: dict = params.get("operational_lead_time", {})
    ro_enabled: bool = bool(ro_cfg.get("enabled", False))
    ro_lead_times: list[int] = [int(h) for h in ro_cfg.get("lead_times", [2, 3])]

    prechampions: list[dict] = prechampion_configs.get("prechampions", [])
    if not prechampions:
        msg = "monthly_prophet_prechampion_configs has no 'prechampions' entries."
        if require_all:
            raise ValueError(msg)
        logger.warning(msg)
        return []

    if test_df.empty:
        raise ValueError("monthly_prophet_test is empty.")
    if target_col not in test_df.columns:
        raise ValueError(
            f"Target column '{target_col}' missing from monthly_prophet_test."
        )

    # Leakage-safe fit window: train + validation only (never the test split).
    fit_df = _concat_sorted([train_df, validation_df], date_col)

    test_df = test_df.copy()
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    test_df = test_df.sort_values(date_col).reset_index(drop=True)

    y_train = train_df[target_col].values.astype(float)  # MASE denominator (train only)
    y_true = test_df[target_col].values.astype(float)
    test_start = str(test_df[date_col].min().date())
    test_end = str(test_df[date_col].max().date())

    rows: list[dict] = []
    for pc in prechampions:
        candidate_id: str = str(pc.get("candidate_id", ""))
        if candidate_id not in candidate_models:
            logger.warning(
                "Prophet candidate '%s' not found in candidate_models — skipping.",
                candidate_id,
            )
            continue

        candidate_model = candidate_models[candidate_id]
        active_regressors: list[str] = list(pc.get("active_regressors", []))

        try:
            refit_model = _refit_prophet_on_frame(
                candidate_model, fit_df, active_regressors, date_col, target_col
            )

            future_df = test_df[[date_col]].rename(columns={date_col: "ds"})
            for reg in active_regressors:
                if reg in test_df.columns:
                    future_df[reg] = test_df[reg].values

            forecast = refit_model.predict(future_df)
            y_pred = forecast["yhat"].values.astype(float)[: len(y_true)]

            # Rolling-origin M-h WAPE (operational lead-time), fresh refit per origin.
            ro_metrics: dict = {}
            if ro_enabled and ro_lead_times:
                ro_metrics = _rolling_origin_wape(
                    fit_df=fit_df,
                    test_df=test_df,
                    target_col=target_col,
                    lead_times=ro_lead_times,
                    forecast_h_step=_make_prophet_h_step(
                        candidate_model, active_regressors, date_col, target_col
                    ),
                )

            rows.append(
                _build_metrics_row(
                    family="prophet",
                    candidate_id=candidate_id,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_train=y_train,
                    mase_period=mase_period,
                    test_start=test_start,
                    test_end=test_end,
                    candidate_rank=pc.get("rank"),
                    extra_metrics=ro_metrics,
                )
            )
            logger.info(
                "Prophet candidate '%s' scored — wape=%.4f%s",
                candidate_id,
                rows[-1]["wape"] or float("nan"),
                _format_rolling_origin_log(ro_metrics),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Prophet candidate '%s' scoring failed: %s", candidate_id, exc
            )

    if not rows and require_all:
        raise RuntimeError(
            "All Prophet prechampion candidates failed during test scoring."
        )
    return rows


def _score_sarimax_candidates(  # noqa: PLR0912, PLR0913, PLR0915
    candidate_models: dict,
    prechampion_configs: dict,
    training_metadata: dict,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    mase_period: int,
    require_all: bool,
) -> list[dict]:
    """Score each SARIMAX prechampion leakage-safely on the held-out test split.

    Each candidate is refit on train + validation only (the test split is never
    seen), then forecast forward over the test period. The MASE denominator uses
    the train-only split, consistent with Prophet scoring.

    Also computes:
    - Ljung-Box residual autocorrelation test and marks candidates with p-value below
      the configured threshold as ``autocorrelation_excluded``.
    - Rolling-origin M-2/M-3 WAPE using progressively extended history windows.
    """
    date_col: str = params.get("date_column", "month_start_date")
    target_col: str = params.get("target_column", "monthly_demand")

    # Ljung-Box config
    ljung_box_cfg: dict = params.get("ljung_box", {})
    lb_enabled: bool = bool(ljung_box_cfg.get("enabled", True))
    lb_lags: int = int(ljung_box_cfg.get("lags", 10))
    lb_threshold: float = float(ljung_box_cfg.get("pvalue_threshold", 0.05))

    # Rolling-origin config
    ro_cfg: dict = params.get("operational_lead_time", {})
    ro_enabled: bool = bool(ro_cfg.get("enabled", True))
    ro_lead_times: list[int] = [int(h) for h in ro_cfg.get("lead_times", [2, 3])]

    candidates_list: list[dict] = prechampion_configs.get("candidates", [])
    if not candidates_list:
        msg = "monthly_sarimax_prechampion_configs has no 'candidates' entries."
        if require_all:
            raise ValueError(msg)
        logger.warning(msg)
        return []

    if test_df.empty:
        raise ValueError("monthly_sarimax_test is empty.")
    if target_col not in test_df.columns:
        raise ValueError(
            f"Target column '{target_col}' missing from monthly_sarimax_test."
        )

    # Fall-back exog source: training metadata (legacy; candidate entry preferred).
    exog_cols_default: list[str] = list(training_metadata.get("exogenous_columns", []))

    # Leakage-safe fit window: train + validation only (never the test split).
    fit_df = _concat_sorted([train_df, validation_df], date_col)

    test_df = test_df.copy()
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    test_df = test_df.sort_values(date_col).reset_index(drop=True)

    y_train = train_df[target_col].values.astype(float)  # MASE denominator (train only)
    y_true = test_df[target_col].values.astype(float)
    fit_y = fit_df[target_col].values.astype(float)
    n_test = len(test_df)

    test_start = str(test_df[date_col].min().date())
    test_end = str(test_df[date_col].max().date())

    rows: list[dict] = []
    for pc in candidates_list:
        trial_id: str = str(pc.get("trial_id", ""))
        if trial_id not in candidate_models:
            logger.warning(
                "SARIMAX candidate '%s' not found in candidate_models — skipping.",
                trial_id,
            )
            continue

        entry = candidate_models[trial_id]
        # Prefer candidate-level exogenous_columns (emitted by hardened training).
        candidate_exog: list[str] = list(
            entry.get("exogenous_columns") or exog_cols_default
        )
        config = entry.get("config", {})
        use_exog: bool = bool(config.get("use_exog", False))

        try:
            # Resolve exog columns present in both fit_df and test_df for consistency.
            available_exog = [
                c
                for c in candidate_exog
                if c in fit_df.columns and c in test_df.columns
            ]
            if use_exog and candidate_exog:
                missing_exog = [
                    c
                    for c in candidate_exog
                    if c not in fit_df.columns or c not in test_df.columns
                ]
                if missing_exog:
                    logger.warning(
                        "SARIMAX candidate '%s': exogenous columns %s not found in "
                        "fit or test DataFrames. Proceeding with: %s",
                        trial_id,
                        missing_exog,
                        available_exog,
                    )
            fit_exog = (
                fit_df[available_exog].values.astype(float)
                if (use_exog and available_exog)
                else None
            )
            test_exog = (
                test_df[available_exog].values.astype(float)
                if (use_exog and available_exog)
                else None
            )

            model_obj = SARIMAX(
                endog=fit_y,
                exog=fit_exog,
                order=tuple(config["order"]),
                seasonal_order=tuple(config["seasonal_order"]),
                trend=config.get("trend"),
                enforce_stationarity=bool(config.get("enforce_stationarity", False)),
                enforce_invertibility=bool(config.get("enforce_invertibility", False)),
            )
            result = model_obj.fit(disp=False)

            # Ljung-Box residual autocorrelation test
            lb_pvalue: float | None = None
            autocorr_excluded: bool = False
            if lb_enabled:
                try:
                    lb_df = acorr_ljungbox(result.resid, lags=[lb_lags], return_df=True)
                    lb_pvalue = float(lb_df["lb_pvalue"].iloc[-1])
                    autocorr_excluded = lb_pvalue < lb_threshold
                    if autocorr_excluded:
                        logger.warning(
                            "SARIMAX candidate '%s': Ljung-Box p-value=%.4f < "
                            "threshold=%.3f — marked for exclusion from champion "
                            "selection.",
                            trial_id,
                            lb_pvalue,
                            lb_threshold,
                        )
                except Exception as lb_exc:  # noqa: BLE001
                    logger.warning(
                        "SARIMAX candidate '%s': Ljung-Box test failed (%s) — "
                        "skipping autocorrelation filter.",
                        trial_id,
                        lb_exc,
                    )

            forecast_out = result.get_forecast(steps=n_test, exog=test_exog)
            y_pred = np.asarray(forecast_out.predicted_mean, dtype=float)[:n_test]

            # Rolling-origin M-2/M-3 metrics
            ro_metrics: dict = {}
            if ro_enabled and ro_lead_times:
                try:
                    ro_metrics = _compute_sarimax_rolling_origin_metrics(
                        fit_df=fit_df,
                        test_df=test_df,
                        config=config,
                        available_exog=available_exog,
                        use_exog=use_exog,
                        date_col=date_col,
                        target_col=target_col,
                        lead_times=ro_lead_times,
                    )
                    logger.info(
                        "SARIMAX candidate '%s' rolling-origin: %s",
                        trial_id,
                        {k: v for k, v in ro_metrics.items() if k.endswith("_wape")},
                    )
                except Exception as ro_exc:  # noqa: BLE001
                    logger.warning(
                        "SARIMAX candidate '%s': rolling-origin metrics failed "
                        "(%s) — skipping.",
                        trial_id,
                        ro_exc,
                    )

            rows.append(
                _build_metrics_row(
                    family="sarimax",
                    candidate_id=trial_id,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_train=y_train,
                    mase_period=mase_period,
                    test_start=test_start,
                    test_end=test_end,
                    candidate_rank=pc.get("rank"),
                    ljung_box_pvalue=lb_pvalue,
                    autocorrelation_excluded=autocorr_excluded,
                    extra_metrics=ro_metrics,
                )
            )
            logger.info(
                "SARIMAX candidate '%s' scored — wape=%.4f  lb_pvalue=%s  "
                "autocorr_excluded=%s",
                trial_id,
                rows[-1]["wape"] or float("nan"),
                f"{lb_pvalue:.4f}" if lb_pvalue is not None else "n/a",
                autocorr_excluded,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SARIMAX candidate '%s' scoring failed: %s", trial_id, exc)

    if not rows and require_all:
        raise RuntimeError(
            "All SARIMAX prechampion candidates failed during test scoring."
        )
    return rows


def _score_catboost_candidates(  # noqa: PLR0912, PLR0913, PLR0915
    candidate_models: dict,
    prechampion_configs: dict,
    split_metadata: dict,
    train_df: pd.DataFrame | None,
    validation_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    params: dict,
    mase_period: int,
    require_all: bool,
) -> list[dict]:
    """Score each CatBoost prechampion leakage-safely on the held-out test split.

    Each candidate is refit on train + validation only (mirroring the Prophet and
    SARIMAX paths), then used to predict over the test period. The test split is
    never seen during fitting. The headline test metrics use a one-shot tabular
    forecast over the precomputed test feature matrix — consistent with how CatBoost
    validation metrics were produced during training (``model.predict(X_val)``). The
    MASE denominator uses the train-only split.

    When ``operational_lead_time`` is enabled, rolling-origin M-h WAPE is also
    computed **recursively** (predicted demand feeds the lag/rolling features of later
    steps, exactly like production inference), so the lead-time metric is honest and
    comparable to the SARIMAX/Prophet M-h numbers rather than an optimistic one-shot
    forecast that uses actual lags.

    Args:
        candidate_models: Dict mapping candidate_id → CatBoost candidate entry dict
            (``config``, ``model``, ``feature_columns``, ``validation_metrics``).
        prechampion_configs: CatBoost prechampion config dict with a ``candidates``
            list (each entry carries ``candidate_id``, ``rank``, ``feature_columns``).
        split_metadata: CatBoost split metadata; supplies fallback column names and
            the ordered feature list (``all_feature_columns``).
        train_df: Training-only split (MASE denominator and first part of the refit
            window).
        validation_df: Validation split appended to train for the leakage-safe refit.
        test_df: Held-out test split with precomputed features and the target column.
        params: Contents of ``model_selection.monthly_catboost`` (column overrides).
        mase_period: Seasonal period for the MASE naive benchmark.
        require_all: When True, raise on missing candidates or universal scoring
            failure; when False, degrade to a warning and skip.

    Returns:
        List of metric row dicts (one per successfully scored candidate).

    Raises:
        ValueError: When required inputs/columns are missing under ``require_all``.
        RuntimeError: When every candidate fails to score under ``require_all``.
    """
    date_col: str = str(
        params.get("date_column")
        or split_metadata.get("date_column", "month_start_date")
    )
    target_col: str = str(
        params.get("target_column")
        or split_metadata.get("target_column", "monthly_demand")
    )

    # Rolling-origin (operational lead-time) config — disabled unless configured.
    ro_cfg: dict = params.get("operational_lead_time", {})
    ro_enabled: bool = bool(ro_cfg.get("enabled", False))
    ro_lead_times: list[int] = [int(h) for h in ro_cfg.get("lead_times", [2, 3])]
    recursive_settings = _resolve_catboost_recursive_settings(split_metadata)

    candidates_list: list[dict] = prechampion_configs.get("candidates", [])
    if not candidates_list:
        msg = "monthly_catboost_prechampion_configs has no 'candidates' entries."
        if require_all:
            raise ValueError(msg)
        logger.warning(msg)
        return []

    if train_df is None or test_df is None:
        msg = (
            "CatBoost is an active family but its train/test splits were not provided "
            "to monthly model selection."
        )
        if require_all:
            raise ValueError(msg)
        logger.warning(msg)
        return []
    if test_df.empty:
        raise ValueError("monthly_catboost_test is empty.")
    if target_col not in test_df.columns:
        raise ValueError(
            f"Target column '{target_col}' missing from monthly_catboost_test."
        )

    default_features: list[str] = list(split_metadata.get("all_feature_columns", []))

    # Leakage-safe fit window: train + validation only (never the test split).
    fit_df = _concat_sorted([train_df, validation_df], date_col)

    test_df = test_df.copy()
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    test_df = test_df.sort_values(date_col).reset_index(drop=True)

    y_train = train_df[target_col].values.astype(float)  # MASE denominator (train only)
    y_true = test_df[target_col].values.astype(float)
    test_start = str(test_df[date_col].min().date())
    test_end = str(test_df[date_col].max().date())

    rows: list[dict] = []
    for pc in candidates_list:
        candidate_id: str = str(pc.get("candidate_id", ""))
        if candidate_id not in candidate_models:
            logger.warning(
                "CatBoost candidate '%s' not found in candidate_models — skipping.",
                candidate_id,
            )
            continue

        entry = candidate_models[candidate_id]
        config: dict = dict(entry.get("config", {}))
        feature_columns: list[str] = list(
            entry.get("feature_columns")
            or pc.get("feature_columns")
            or default_features
        )
        fitted_model = entry.get("model")

        try:
            # Resolve features present in both the fit window and the test split.
            available = [
                c
                for c in feature_columns
                if c in fit_df.columns and c in test_df.columns
            ]
            if not available:
                raise ValueError(
                    "No CatBoost feature columns are present in both the fit and "
                    "test frames; cannot build the feature matrix."
                )
            missing = [c for c in feature_columns if c not in available]
            if missing:
                logger.warning(
                    "CatBoost candidate '%s': feature columns %s missing from fit or "
                    "test frame. Proceeding with %d available column(s).",
                    candidate_id,
                    missing,
                    len(available),
                )

            x_fit = fit_df[available].to_numpy(dtype=float)
            y_fit = fit_df[target_col].to_numpy(dtype=float)
            x_test = test_df[available].to_numpy(dtype=float)

            refit_model = _refit_catboost_on_arrays(config, x_fit, y_fit, fitted_model)
            y_pred = np.asarray(refit_model.predict(x_test), dtype=float)[: len(y_true)]

            # Recursive rolling-origin M-h WAPE (operational lead-time), fresh refit
            # per origin with predicted demand feeding later lag/rolling features.
            ro_metrics: dict = {}
            if ro_enabled and ro_lead_times:
                ro_metrics = _rolling_origin_wape(
                    fit_df=fit_df,
                    test_df=test_df,
                    target_col=target_col,
                    lead_times=ro_lead_times,
                    forecast_h_step=_make_catboost_h_step(
                        config=config,
                        fitted_model=fitted_model,
                        feature_columns=available,
                        target_col=target_col,
                        date_col=date_col,
                        recursive_settings=recursive_settings,
                    ),
                )

            rows.append(
                _build_metrics_row(
                    family="catboost",
                    candidate_id=candidate_id,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_train=y_train,
                    mase_period=mase_period,
                    test_start=test_start,
                    test_end=test_end,
                    candidate_rank=pc.get("rank"),
                    extra_metrics=ro_metrics,
                )
            )
            logger.info(
                "CatBoost candidate '%s' scored — wape=%.4f%s",
                candidate_id,
                rows[-1]["wape"] or float("nan"),
                _format_rolling_origin_log(ro_metrics),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CatBoost candidate '%s' scoring failed: %s", candidate_id, exc
            )

    if not rows and require_all:
        raise RuntimeError(
            "All CatBoost prechampion candidates failed during test scoring."
        )
    return rows


def _refit_catboost_on_arrays(
    config: dict,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    fitted_model: Any | None = None,
) -> CatBoostRegressor:
    """Rebuild a CatBoostRegressor from a candidate config and fit it on ``x_fit``.

    Reproduces the selected configuration on a new data window (train+validation for
    test scoring, or full history for the champion). No ``eval_set`` is supplied —
    the held-out window must never leak into fitting — so early stopping is not used.
    When the original fitted model is available, the tree count is capped at its
    early-stopped best iteration so the refit reproduces the selected complexity
    rather than the full ``iterations`` ceiling.

    Args:
        config: Hyperparameter dict from the candidate entry (valid CatBoost kwargs).
        x_fit: Feature matrix for fitting.
        y_fit: Target array for fitting.
        fitted_model: Original train-only fitted model (used to read the best
            iteration). May be None.

    Returns:
        A freshly fitted CatBoostRegressor.
    """
    fit_config = {k: v for k, v in config.items() if k != "random_seed"}

    best_iteration: int | None = None
    if fitted_model is not None:
        try:
            best_iteration = int(fitted_model.get_best_iteration())
        except Exception:  # noqa: BLE001
            best_iteration = None
    if best_iteration and best_iteration > 0:
        fit_config["iterations"] = best_iteration + 1

    model = CatBoostRegressor(**fit_config, verbose=False, allow_writing_files=False)
    model.fit(x_fit, y_fit, verbose=False)
    return model


def _resolve_catboost_recursive_settings(split_metadata: dict) -> dict:
    """Resolve the recursive target-feature settings from CatBoost split metadata.

    Mirrors the resolution performed by the production CatBoost inference adapter so
    the rolling-origin recursion reproduces exactly how the champion forecasts.

    Returns:
        Dict with ``target_lags``, ``rolling_windows``, ``include_std``,
        ``include_min_max``, ``trend_diffs``, and ``trend_pct_changes``.
    """
    lag_settings = dict(split_metadata.get("lag_settings") or {})
    target_lags = sorted(
        int(x) for x in lag_settings.get("target_lags", [1, 2, 3, 6, 12])
    )

    rolling_settings = dict(split_metadata.get("rolling_settings") or {})
    rolling_windows = sorted(
        int(x) for x in rolling_settings.get("windows", [3, 6, 12])
    )
    include_std = bool(rolling_settings.get("include_std", True))
    include_min_max = bool(rolling_settings.get("include_min_max", True))

    trend_feature_cols = list(split_metadata.get("trend_feature_columns") or [])
    trend_diffs = extract_periods_from_column_names(trend_feature_cols, "demand_diff_")
    trend_pct_changes = extract_periods_from_column_names(
        trend_feature_cols, "demand_pct_change_"
    )

    return {
        "target_lags": target_lags,
        "rolling_windows": rolling_windows,
        "include_std": include_std,
        "include_min_max": include_min_max,
        "trend_diffs": trend_diffs,
        "trend_pct_changes": trend_pct_changes,
    }


def _make_catboost_h_step(  # noqa: PLR0913
    config: dict,
    fitted_model: Any | None,
    feature_columns: list[str],
    target_col: str,
    date_col: str,
    recursive_settings: dict,
) -> Callable[[pd.DataFrame, pd.DataFrame, int], float]:
    """Build a recursive rolling-origin h-step forecast function for a CatBoost candidate.

    The returned callable refits CatBoost on the origin window, seeds the demand
    buffer from the origin's observed demand, then forecasts ``h`` steps recursively —
    each prediction is appended to the buffer so it feeds the lag/rolling/trend
    features of subsequent steps, exactly like production inference. Future-known
    calendar/exogenous features come from the corresponding test rows.

    Args:
        config: CatBoost hyperparameter dict from the candidate entry.
        fitted_model: Train-only fitted model (used to cap the tree count).
        feature_columns: Ordered feature columns to build the prediction matrix.
        target_col: Target column name.
        date_col: Date column name.
        recursive_settings: Output of ``_resolve_catboost_recursive_settings``.

    Returns:
        Callable ``(origin_df, future_slice, h) -> float``.
    """

    def _h_step(origin_df: pd.DataFrame, future_slice: pd.DataFrame, h: int) -> float:
        x_origin = origin_df[feature_columns].to_numpy(dtype=float)
        y_origin = origin_df[target_col].to_numpy(dtype=float)
        model = _refit_catboost_on_arrays(config, x_origin, y_origin, fitted_model)

        demand_buffer = build_demand_buffer(origin_df, target_col, date_col)
        future_sorted = future_slice.sort_values(date_col)

        predictions: list[float] = []
        for _, row in future_sorted.iterrows():
            recursive_feats = compute_recursive_target_features(
                demand_buffer=demand_buffer,
                target_lags=recursive_settings["target_lags"],
                rolling_windows=recursive_settings["rolling_windows"],
                include_std=recursive_settings["include_std"],
                include_min_max=recursive_settings["include_min_max"],
                trend_diffs=recursive_settings["trend_diffs"],
                trend_pct_changes=recursive_settings["trend_pct_changes"],
            )
            feature_values = [
                (
                    recursive_feats[col]
                    if col in recursive_feats
                    else (row[col] if col in row.index else np.nan)
                )
                for col in feature_columns
            ]
            prediction = float(
                model.predict(np.array([feature_values], dtype=float))[0]
            )
            predictions.append(prediction)
            demand_buffer.append(prediction)

        return predictions[h - 1]

    return _h_step


def _build_metrics_row(  # noqa: PLR0913
    family: str,
    candidate_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    mase_period: int,
    test_start: str,
    test_end: str,
    candidate_rank: int | None,
    ljung_box_pvalue: float | None = None,
    autocorrelation_excluded: bool = False,
    extra_metrics: dict | None = None,
) -> dict:
    """Compute test metrics and return a single candidate row dict."""
    wape_val = _safe_float(_shared_wape(y_true, y_pred))
    rmse_val = _safe_float(_shared_rmse(y_true, y_pred))

    mase_val: float | None = None
    if len(y_train) > mase_period:
        raw = _shared_mase(y_true, y_pred, y_train, mase_period)
        mase_val = _safe_float(raw)

    epsilon = 1.0
    bias_val = _safe_float(
        float(np.sum(y_pred - y_true)) / (float(np.sum(np.abs(y_true))) + epsilon)
    )

    row = {
        "family": family,
        "candidate_id": candidate_id,
        "candidate_rank": candidate_rank,
        "granularity": "monthly",
        "selection_stage": "test",
        "test_start_date": test_start,
        "test_end_date": test_end,
        "n_test_rows": len(y_true),
        "wape": wape_val,
        "mase": mase_val,
        "rmse": rmse_val,
        "bias": bias_val,
        "ljung_box_pvalue": ljung_box_pvalue,
        "autocorrelation_excluded": autocorrelation_excluded,
        "primary_metric": "wape",
        "primary_metric_value": wape_val,
        "is_family_champion": False,
        "is_production_champion": False,
    }
    if extra_metrics:
        row.update(extra_metrics)
    return row


def _compute_sarimax_rolling_origin_metrics(  # noqa: PLR0913
    fit_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    available_exog: list[str],
    use_exog: bool,
    date_col: str,
    target_col: str,
    lead_times: list[int],
) -> dict:
    """Compute rolling-origin WAPE for each lead time using the held-out test window.

    For lead_time h, the set of valid (origin, target) pairs is:
    - target = test_df row at 0-based index i, where i >= h - 1
    - origin = fit_df extended with the first (i - h + 1) test rows

    Each origin gets a fresh SARIMAX refit; the h-th step forecast is compared to
    the actual at position i. The pairs are aggregated into WAPE.

    Args:
        fit_df: Combined train+validation DataFrame (leakage-safe fit window).
        test_df: Held-out test DataFrame already sorted by date_col.
        config: SARIMAX order/seasonal_order/trend/enforce_* config dict.
        available_exog: Exogenous column names available in both fit_df and test_df.
        use_exog: Whether to pass exogenous features to SARIMAX.
        date_col: Date column name.
        target_col: Target column name.
        lead_times: List of lead-time horizons to evaluate (e.g. [2, 3]).

    Returns:
        Dict with keys ``test_m{h}_wape`` and ``n_m{h}_pairs`` for each ``h``
        in ``lead_times``. Values are ``None`` / 0 when no valid pairs exist.
    """
    results: dict = {}

    for h in lead_times:
        actuals: list[float] = []
        preds: list[float] = []

        for i in range(len(test_df)):
            if i < h - 1:
                continue  # not enough history ahead of this target for lead_time h

            n_extra = i - (h - 1)  # test rows to extend the origin window
            if n_extra == 0:
                origin_df = fit_df
            else:
                origin_df = pd.concat(
                    [fit_df, test_df.iloc[:n_extra].copy()], ignore_index=True
                )

            origin_y = origin_df[target_col].values.astype(float)
            origin_exog = (
                origin_df[available_exog].values.astype(float)
                if (use_exog and available_exog)
                else None
            )
            future_slice = test_df.iloc[n_extra : n_extra + h]
            future_exog = (
                future_slice[available_exog].values.astype(float)
                if (use_exog and available_exog)
                else None
            )

            try:
                model_ro = SARIMAX(
                    endog=origin_y,
                    exog=origin_exog,
                    order=tuple(config["order"]),
                    seasonal_order=tuple(config["seasonal_order"]),
                    trend=config.get("trend"),
                    enforce_stationarity=bool(
                        config.get("enforce_stationarity", False)
                    ),
                    enforce_invertibility=bool(
                        config.get("enforce_invertibility", False)
                    ),
                )
                with _warnings.catch_warnings(record=True):
                    _warnings.simplefilter("always")
                    result_ro = model_ro.fit(disp=False)

                forecast_ro = result_ro.get_forecast(steps=h, exog=future_exog)
                y_pred_ro = np.asarray(forecast_ro.predicted_mean, dtype=float)

                if len(y_pred_ro) >= h:
                    pred_h = float(y_pred_ro[h - 1])
                    actual_h = float(test_df.iloc[i][target_col])
                    if np.isfinite(pred_h) and np.isfinite(actual_h):
                        preds.append(pred_h)
                        actuals.append(actual_h)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Rolling-origin SARIMAX: refit failed at test index=%d lead_time=%d "
                    "— skipping pair.",
                    i,
                    h,
                )
                continue

        m_label = f"test_m{h}_wape"
        n_label = f"n_m{h}_pairs"
        if actuals:
            results[m_label] = _safe_float(
                _shared_wape(np.array(actuals), np.array(preds))
            )
            results[n_label] = len(actuals)
        else:
            results[m_label] = None
            results[n_label] = 0

    return results


def _rolling_origin_wape(
    fit_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    lead_times: list[int],
    forecast_h_step: Callable[[pd.DataFrame, pd.DataFrame, int], float],
) -> dict:
    """Compute rolling-origin (operational lead-time) WAPE for each lead time.

    For lead_time ``h`` the valid (origin, target) pairs are:
    - target = ``test_df`` row at 0-based index ``i``, where ``i >= h - 1``
    - origin = ``fit_df`` extended with the first ``i - h + 1`` test rows

    ``forecast_h_step(origin_df, future_slice, h)`` must refit the model on the origin
    window and return the ``h``-step-ahead point forecast for the target row. Because
    a fresh model is fit at each origin, the metric reflects genuine multi-step
    extrapolation. The pairing mirrors ``_compute_sarimax_rolling_origin_metrics`` so
    M-h metrics are directly comparable across Prophet, SARIMAX, and CatBoost.

    Args:
        fit_df: Combined train+validation DataFrame (leakage-safe fit window).
        test_df: Held-out test DataFrame already sorted by date.
        target_col: Target column name.
        lead_times: Lead-time horizons to evaluate (e.g. ``[2, 3]``).
        forecast_h_step: Callable returning the h-step-ahead forecast for the target.

    Returns:
        Dict with keys ``test_m{h}_wape`` and ``n_m{h}_pairs`` for each ``h``.
    """
    results: dict = {}
    for h in lead_times:
        actuals: list[float] = []
        preds: list[float] = []
        for i in range(len(test_df)):
            if i < h - 1:
                continue  # not enough test history ahead of this target for lead h
            n_extra = i - (h - 1)
            origin_df = (
                fit_df
                if n_extra == 0
                else pd.concat([fit_df, test_df.iloc[:n_extra]], ignore_index=True)
            )
            future_slice = test_df.iloc[n_extra : n_extra + h]
            try:
                pred_h = float(forecast_h_step(origin_df, future_slice, h))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Rolling-origin forecast failed at test index=%d lead_time=%d: %s",
                    i,
                    h,
                    exc,
                )
                continue
            actual_h = float(test_df.iloc[i][target_col])
            if np.isfinite(pred_h) and np.isfinite(actual_h):
                preds.append(pred_h)
                actuals.append(actual_h)

        if actuals:
            results[f"test_m{h}_wape"] = _safe_float(
                _shared_wape(np.array(actuals), np.array(preds))
            )
            results[f"n_m{h}_pairs"] = len(actuals)
        else:
            results[f"test_m{h}_wape"] = None
            results[f"n_m{h}_pairs"] = 0
    return results


def _format_rolling_origin_log(ro_metrics: dict) -> str:
    """Format rolling-origin M-h WAPE values as a one-line log suffix."""
    if not ro_metrics:
        return ""
    parts = [
        f"{k}={v:.4f}"
        for k, v in sorted(ro_metrics.items())
        if k.endswith("_wape") and v is not None
    ]
    return f"  rolling-origin: {', '.join(parts)}" if parts else ""


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

    The CatBoost champion artifact from the direct E1 training node carries three
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


def _make_prophet_h_step(
    candidate_model: Any,
    active_regressors: list[str],
    date_col: str,
    target_col: str,
) -> Callable[[pd.DataFrame, pd.DataFrame, int], float]:
    """Build a rolling-origin h-step forecast function for a Prophet candidate.

    The returned callable refits the candidate's configuration on the origin window
    and returns the h-step-ahead ``yhat`` for the target row, so the operational
    lead-time WAPE reflects a genuine multi-step Prophet forecast.

    Args:
        candidate_model: The fitted Prophet candidate (source of hyperparameters).
        active_regressors: Regressor columns the candidate uses.
        date_col: Date column name in the origin/future frames.
        target_col: Target column name in the origin frame.

    Returns:
        Callable ``(origin_df, future_slice, h) -> float``.
    """

    def _h_step(origin_df: pd.DataFrame, future_slice: pd.DataFrame, h: int) -> float:
        ro_model = _refit_prophet_on_frame(
            candidate_model, origin_df, active_regressors, date_col, target_col
        )
        future = future_slice[[date_col]].rename(columns={date_col: "ds"})
        for reg in active_regressors:
            if reg in future_slice.columns:
                future[reg] = future_slice[reg].to_numpy()
        forecast = ro_model.predict(future)
        return float(forecast["yhat"].to_numpy()[h - 1])

    return _h_step


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
        # CatBoost direct multi-horizon: max 3 months (protocol §8).
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
