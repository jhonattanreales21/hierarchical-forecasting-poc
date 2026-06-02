"""Monthly multi-family model selection nodes.

Compares prechampion candidates on the held-out monthly test
period, selects one family champion per family, then elects one monthly production
champion.
"""

import logging
import warnings as _warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from shared.metrics import mase as _shared_mase
from shared.metrics import rmse as _shared_rmse
from shared.metrics import wape as _shared_wape

logger = logging.getLogger(__name__)


# ── Node 1 ────────────────────────────────────────────────────────────────────


def evaluate_monthly_family_candidates_on_test(
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
) -> pd.DataFrame:
    """Score all Prophet and SARIMAX prechampion candidates on the held-out test set.

    Evaluates only the families listed in ``params_monthly.active_families``. Both
    families are scored consistently: each candidate is refit on the combined
    train+validation split and then forecast forward over the held-out test period.
    The test split is never seen during fitting, so the reported test metrics are
    leakage-safe and the Prophet-vs-SARIMAX comparison is fair (both see the same
    data at test time). A full-history refit (including the test split) happens only
    later, for the elected production champion.

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


def select_monthly_family_champions(
    monthly_candidate_test_metrics: pd.DataFrame,
    params_monthly: dict,
) -> pd.DataFrame:
    """Select the best test-period candidate within each active model family.

    Ranks candidates by the configured primary metric (default: wape) and applies
    tie-breakers in order. Exactly one family champion is selected per active
    family present in ``monthly_candidate_test_metrics``.

    Args:
        monthly_candidate_test_metrics: Output of
            ``evaluate_monthly_family_candidates_on_test``.
        params_monthly: Contents of ``model_selection.monthly``; must contain
            ``primary_metric`` and ``tie_breakers``.

    Returns:
        DataFrame with one row per family champion containing: family, granularity,
        family_champion_id, family_champion_rank, wape, mase, rmse, bias,
        selection_reason, model_artifact_key, metadata_artifact_key.

    Raises:
        ValueError: When the input metrics table is empty or missing required columns.
    """
    if monthly_candidate_test_metrics.empty:
        raise ValueError(
            "monthly_candidate_test_metrics is empty; cannot select family champions."
        )

    primary_metric: str = str(params_monthly.get("primary_metric", "wape"))
    tie_breakers: list[str] = list(
        params_monthly.get("tie_breakers", ["mase", "rmse", "abs_bias"])
    )
    active_families: list[str] = sorted(
        monthly_candidate_test_metrics["family"].unique().tolist()
    )

    champion_rows: list[dict] = []
    for family in active_families:
        family_df = monthly_candidate_test_metrics[
            monthly_candidate_test_metrics["family"] == family
        ].copy()

        if family_df.empty:
            logger.warning("No candidates found for family '%s' — skipping.", family)
            continue

        # For SARIMAX: exclude candidates flagged by the Ljung-Box filter before ranking.
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
                "wape": _safe_float(best.get("wape")),
                "mase": _safe_float(best.get("mase")),
                "rmse": _safe_float(best.get("rmse")),
                "bias": _safe_float(best.get("bias")),
                "selection_reason": (
                    f"Best {primary_metric} among {len(family_df)} {family} candidate(s) "
                    f"on held-out test set"
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
    monthly_candidate_test_metrics: pd.DataFrame,
    params_monthly: dict,
) -> pd.DataFrame:
    """Compare family champions and elect one monthly production champion.

    Only family champions (one per family) participate in the final comparison.
    The selection uses the same metric ordering as family champion selection:
    primary metric first, then tie-breakers in order.

    Args:
        monthly_family_champion_summary: Output of ``select_monthly_family_champions``.
        monthly_candidate_test_metrics: Full candidate metrics table for summary stats.
        params_monthly: Contents of ``model_selection.monthly``.

    Returns:
        Single-row DataFrame describing the production champion selection:
        granularity, active_families, production_champion_family,
        production_champion_id, primary_metric, primary_metric_value,
        tie_breakers, selection_timestamp, selection_reason, candidate_count,
        family_champion_count.

    Raises:
        ValueError: When the family champion summary is empty.
    """
    if monthly_family_champion_summary.empty:
        raise ValueError(
            "monthly_family_champion_summary is empty; cannot select production champion."
        )

    primary_metric: str = str(params_monthly.get("primary_metric", "wape"))
    tie_breakers: list[str] = list(
        params_monthly.get("tie_breakers", ["mase", "rmse", "abs_bias"])
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
        "candidate_count": len(monthly_candidate_test_metrics),
        "family_champion_count": len(ranked),
    }
    return pd.DataFrame([summary_row])


# ── Node 4 ────────────────────────────────────────────────────────────────────


def build_monthly_champion_artifacts(  # noqa: PLR0913
    monthly_model_selection_summary: pd.DataFrame,
    monthly_family_champion_summary: pd.DataFrame,
    monthly_candidate_test_metrics: pd.DataFrame,
    monthly_prophet_candidate_models: dict,
    monthly_sarimax_candidate_models: dict,
    monthly_prophet_full_train: pd.DataFrame,
    monthly_sarimax_full_train: pd.DataFrame,
    monthly_sarimax_training_metadata: dict,
    params_monthly: dict,
) -> tuple[Any, dict]:
    """Refit the production champion on full history and build its JSON metadata.

    Champion Protocol stage 5 (full-history refit): the elected champion
    configuration is refit on all available history (train + validation + test) so
    that production inference benefits from the full training window. Reported test
    metrics are unchanged — they come from the pre-refit selection stage. When
    ``refit_champion.enabled`` is false, the train-only candidate is returned as-is.

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

    Returns:
        Two-element tuple:

        1. ``champion_monthly_model`` — the production champion model, refit on full
           history (Prophet model, or SARIMAX candidate dict carrying the refit
           results object under ``"model"``).
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
    )

    champion_model, inference_contract, refit_info = _build_production_champion_model(
        production_family=production_family,
        candidate_model=candidate_model,
        prophet_full_train=monthly_prophet_full_train,
        sarimax_full_train=monthly_sarimax_full_train,
        sarimax_training_metadata=monthly_sarimax_training_metadata,
        params_monthly=params_monthly,
    )

    champ_metrics = _extract_champion_metrics(
        monthly_candidate_test_metrics, production_family, production_candidate_id
    )

    family_champions: dict[str, dict] = {}
    for _, fc_row in monthly_family_champion_summary.iterrows():
        fam = str(fc_row["family"])
        family_champions[fam] = {
            "champion_id": str(fc_row["family_champion_id"]),
            "wape": _safe_float(fc_row.get("wape")),
            "mase": _safe_float(fc_row.get("mase")),
            "rmse": _safe_float(fc_row.get("rmse")),
            "bias": _safe_float(fc_row.get("bias")),
        }

    test_start, test_end, n_rows = _extract_test_period(monthly_candidate_test_metrics)

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
            "primary_metric": str(summary_row.get("primary_metric", "wape")),
            "direction": "minimize",
            "tie_breakers": list(
                params_monthly.get("tie_breakers", ["mase", "rmse", "abs_bias"])
            ),
            "selected_at": str(
                summary_row.get("selection_timestamp", datetime.now(tz=UTC).isoformat())
            ),
            "selection_reason": str(summary_row.get("selection_reason", "")),
        },
        "test_period": {
            "start_date": test_start,
            "end_date": test_end,
            "n_rows": n_rows,
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

    logger.info(
        "champion_monthly_model built — family=%s  candidate=%s  refit=%s (n_obs=%s)",
        production_family,
        production_candidate_id,
        refit_info["performed"],
        refit_info["n_obs"],
    )
    return champion_model, metadata


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
    if missing and require_all:
        raise ValueError(
            "require_all_active_families=true but the following families have no "
            f"prechampion artifacts: {missing}"
        )
    for msg in missing:
        logger.warning("Active family missing artifacts: %s", msg)


def _score_prophet_candidates(
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
    """
    date_col: str = params.get("date_column", "ds")
    target_col: str = params.get("target_column", "y")

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
                )
            )
            logger.info(
                "Prophet candidate '%s' scored — wape=%.4f",
                candidate_id,
                rows[-1]["wape"] or float("nan"),
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


def _score_sarimax_candidates(  # noqa: PLR0915
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
        candidate_exog: list[str] = list(entry.get("exogenous_columns") or exog_cols_default)
        config = entry.get("config", {})
        use_exog: bool = bool(config.get("use_exog", False))

        try:
            # Resolve exog columns present in both fit_df and test_df for consistency.
            available_exog = [
                c for c in candidate_exog if c in fit_df.columns and c in test_df.columns
            ]
            if use_exog and candidate_exog:
                missing_exog = [
                    c for c in candidate_exog
                    if c not in fit_df.columns or c not in test_df.columns
                ]
                if missing_exog:
                    logger.warning(
                        "SARIMAX candidate '%s': exogenous columns %s not found in "
                        "fit or test DataFrames. Proceeding with: %s",
                        trial_id, missing_exog, available_exog,
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
                            trial_id, lb_pvalue, lb_threshold,
                        )
                except Exception as lb_exc:  # noqa: BLE001
                    logger.warning(
                        "SARIMAX candidate '%s': Ljung-Box test failed (%s) — "
                        "skipping autocorrelation filter.",
                        trial_id, lb_exc,
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
                        trial_id, ro_exc,
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
                    enforce_stationarity=bool(config.get("enforce_stationarity", False)),
                    enforce_invertibility=bool(config.get("enforce_invertibility", False)),
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
                    i, h,
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
    raise ValueError(
        f"Unknown production champion family '{production_family}'. "
        "Expected 'prophet' or 'sarimax'."
    )


def _build_production_champion_model(  # noqa: PLR0913
    production_family: str,
    candidate_model: Any,
    prophet_full_train: pd.DataFrame,
    sarimax_full_train: pd.DataFrame,
    sarimax_training_metadata: dict,
    params_monthly: dict,
) -> tuple[Any, dict, dict]:
    """Refit the elected champion on full history and build its inference contract.

    Implements Champion Protocol stage 5. The winning configuration is refit on all
    available history so production forecasts use the full training window. When
    ``refit_champion.enabled`` is false, the train-only candidate is returned
    unchanged (useful for fast debugging runs).

    Args:
        production_family: Elected production champion family.
        candidate_model: Train-only champion artifact resolved from the candidate
            pool (Prophet model or SARIMAX candidate dict).
        prophet_full_train: Prophet-format full history (ds, y, regressors).
        sarimax_full_train: SARIMAX-format full history (date, target, exog).
        sarimax_training_metadata: SARIMAX training metadata (exogenous_columns).
        params_monthly: Contents of ``model_selection.monthly``.

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
        contract = _build_inference_contract("sarimax", used_exog, sarimax_config=config)
        return champion_model, contract, refit_info

    raise ValueError(
        f"Cannot build a production champion model for unknown family "
        f"'{production_family}'."
    )


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
        mode = extra[name].get("mode", "additive") if isinstance(extra.get(name), dict) else "additive"
        model.add_regressor(name, mode=mode)
    model.fit(fit_input)
    return model


def _extract_prophet_params(model: Any) -> dict:
    """Extract Prophet constructor hyperparameters from a fitted model (with defaults)."""
    return {
        "changepoint_prior_scale": float(getattr(model, "changepoint_prior_scale", 0.05)),
        "seasonality_prior_scale": float(getattr(model, "seasonality_prior_scale", 10.0)),
        "holidays_prior_scale": float(getattr(model, "holidays_prior_scale", 10.0)),
        "seasonality_mode": str(getattr(model, "seasonality_mode", "additive")),
        "yearly_seasonality": getattr(model, "yearly_seasonality", True),
        "weekly_seasonality": getattr(model, "weekly_seasonality", False),
        "daily_seasonality": getattr(model, "daily_seasonality", False),
        "interval_width": float(getattr(model, "interval_width", 0.8)),
    }


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
    dates = pd.to_datetime(full_train_df[date_col]) if date_col in full_train_df.columns else None
    return {
        "performed": performed,
        "data_scope": "full_history",  # train + validation + test
        "n_obs": int(len(full_train_df)),
        "start_date": str(dates.min().date()) if dates is not None and not dates.empty else None,
        "end_date": str(dates.max().date()) if dates is not None and not dates.empty else None,
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
        return _extract_prophet_params(champion_model)
    if production_family == "sarimax":
        config = dict(inference_contract.get("sarimax_config") or {})
        return {
            "order": config.get("order"),
            "seasonal_order": config.get("seasonal_order"),
            "trend": config.get("trend"),
            "use_exog": bool(config.get("use_exog", False)),
        }
    return {}


def _build_inference_contract(
    production_family: str,
    active_regressors: list[str],
    sarimax_config: dict | None = None,
) -> dict:
    """Describe how the champion must be consumed at inference time.

    The contract is self-describing so metadata-driven inference does not need to
    introspect the model object. ``active_regressors`` reflects the exact features
    the (refit) champion was fit with; column names follow each family's convention.

    Args:
        production_family: Elected production champion family.
        active_regressors: Regressor/exogenous column names the champion uses.
        sarimax_config: SARIMAX order configuration, when applicable.

    Returns:
        Dict describing column conventions, active regressors, interval support, and
        (for SARIMAX) the model order configuration.
    """
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
                "enforce_invertibility": bool(config.get("enforce_invertibility", False)),
            },
        }

    raise ValueError(
        f"Cannot build an inference contract for unknown family '{production_family}'."
    )


def _extract_champion_metrics(
    metrics_df: pd.DataFrame, family: str, candidate_id: str
) -> dict:
    """Extract scalar metrics for the production champion from the metrics table."""
    mask = (metrics_df["family"] == family) & (
        metrics_df["candidate_id"] == candidate_id
    )
    subset = metrics_df[mask]
    if subset.empty:
        return {"wape": None, "mase": None, "rmse": None, "bias": None}
    row = subset.iloc[0]
    return {
        "wape": _safe_float(row.get("wape")),
        "mase": _safe_float(row.get("mase")),
        "rmse": _safe_float(row.get("rmse")),
        "bias": _safe_float(row.get("bias")),
    }


def _extract_test_period(metrics_df: pd.DataFrame) -> tuple[str, str, int]:
    """Return (test_start, test_end, n_rows) from the metrics table."""
    if metrics_df.empty:
        return ("", "", 0)
    return (
        str(metrics_df["test_start_date"].iloc[0]),
        str(metrics_df["test_end_date"].iloc[0]),
        int(metrics_df["n_test_rows"].iloc[0]),
    )


def _safe_float(value: Any) -> float | None:
    """Convert to Python float, returning None for missing or non-finite values."""
    if value is None:
        return None
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None
