"""Monthly multi-family model selection nodes.

Compares prechampion candidates on the held-out monthly test
period, selects one family champion per family, then elects one monthly production
champion.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
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
    monthly_prophet_test: pd.DataFrame,
    monthly_sarimax_candidate_models: dict,
    monthly_sarimax_prechampion_configs: dict,
    monthly_sarimax_training_metadata: dict,
    monthly_sarimax_train: pd.DataFrame,
    monthly_sarimax_test: pd.DataFrame,
    monthly_sarimax_full_train: pd.DataFrame,
    params_monthly: dict,
    params_prophet: dict,
    params_sarimax: dict,
) -> pd.DataFrame:
    """Score all Prophet and SARIMAX prechampion candidates on the held-out test set.

    Evaluates only the families listed in ``params_monthly.active_families``. For
    each Prophet candidate the existing fitted model is used to forecast the test
    dates directly. For each SARIMAX candidate the model is refitted on the
    combined train+validation split (``monthly_sarimax_full_train``) so that the
    forecast horizon aligns with the test period without requiring iterated
    multi-step forecasts through the validation window.

    Args:
        monthly_prophet_candidate_models: Dict mapping candidate_id → fitted
            Prophet model from the training stage.
        monthly_prophet_prechampion_configs: Prechampion config dict emitted by
            Prophet training, containing a ``prechampions`` list.
        monthly_prophet_train: Training-only split used as MASE denominator for
            Prophet candidates (no validation or test rows, leakage-safe).
        monthly_prophet_test: Held-out Prophet test split with ds, y, and regressor
            columns.
        monthly_sarimax_candidate_models: Dict mapping trial_id → SARIMAX candidate
            entry (rank, config, model, validation_metrics).
        monthly_sarimax_prechampion_configs: Prechampion config dict emitted by
            SARIMAX training, containing a ``candidates`` list.
        monthly_sarimax_training_metadata: Training metadata dict; used to retrieve
            the exogenous column names used during training.
        monthly_sarimax_train: Training-only split used as MASE denominator for
            SARIMAX candidates.
        monthly_sarimax_test: Held-out SARIMAX test split with date, target, and
            optional exogenous columns.
        monthly_sarimax_full_train: Combined train+validation split for SARIMAX
            refit before test scoring.
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
            test_df=monthly_sarimax_test,
            full_train_df=monthly_sarimax_full_train,
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


def build_monthly_champion_artifacts(
    monthly_model_selection_summary: pd.DataFrame,
    monthly_family_champion_summary: pd.DataFrame,
    monthly_candidate_test_metrics: pd.DataFrame,
    monthly_prophet_candidate_models: dict,
    monthly_sarimax_candidate_models: dict,
    params_monthly: dict,
) -> tuple[Any, dict]:
    """Retrieve the production champion model and build its JSON metadata.

    Resolves the elected champion from ``monthly_model_selection_summary`` to the
    actual model object and constructs a JSON-serialisable metadata dict with the
    full selection audit trail.

    Args:
        monthly_model_selection_summary: Single-row summary from
            ``select_monthly_production_champion``.
        monthly_family_champion_summary: Family champions table from
            ``select_monthly_family_champions``.
        monthly_candidate_test_metrics: Full candidate metrics table.
        monthly_prophet_candidate_models: Dict mapping candidate_id → Prophet model.
        monthly_sarimax_candidate_models: Dict mapping trial_id → SARIMAX candidate
            entry dict (including the ``model`` key).
        params_monthly: Contents of ``model_selection.monthly``.

    Returns:
        Two-element tuple:

        1. ``champion_monthly_model`` — the model object for the elected champion.
        2. ``champion_monthly_metadata`` — JSON-serialisable dict describing the
           champion, metrics, test period, and compatibility notes.

    Raises:
        ValueError: When the summary is empty or the champion family is unknown.
        RuntimeError: When the champion candidate ID cannot be resolved to a model.
    """
    if monthly_model_selection_summary.empty:
        raise ValueError("monthly_model_selection_summary is empty.")

    summary_row = monthly_model_selection_summary.iloc[0]
    production_family: str = str(summary_row["production_champion_family"])
    production_candidate_id: str = str(summary_row["production_champion_id"])

    champion_model = _resolve_champion_model(
        production_family=production_family,
        production_candidate_id=production_candidate_id,
        prophet_candidate_models=monthly_prophet_candidate_models,
        sarimax_candidate_models=monthly_sarimax_candidate_models,
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
        "model_artifact": {
            "catalog_key": "champion_monthly_model",
            "source_candidate_key": f"monthly_{production_family}_candidate_models",
            "source_candidate_id": production_candidate_id,
        },
        "compatibility": {
            "legacy_prophet_artifacts_preserved": True,
            "inference_ready": False,
            "requires_phase_6_metadata_driven_inference": True,
        },
    }

    logger.info(
        "champion_monthly_model resolved — family=%s  candidate=%s",
        production_family,
        production_candidate_id,
    )
    return champion_model, metadata


# ── Private helpers ───────────────────────────────────────────────────────────


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
    test_df: pd.DataFrame,
    params: dict,
    mase_period: int,
    require_all: bool,
) -> list[dict]:
    """Score each Prophet prechampion on the held-out test set."""
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

    test_df = test_df.copy()
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    test_df = test_df.sort_values(date_col).reset_index(drop=True)

    y_train = train_df[target_col].values.astype(float)
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

        model = candidate_models[candidate_id]
        active_regressors: list[str] = list(pc.get("active_regressors", []))

        try:
            future_df = test_df[[date_col]].rename(columns={date_col: "ds"})
            for reg in active_regressors:
                if reg in test_df.columns:
                    future_df[reg] = test_df[reg].values

            forecast = model.predict(future_df)
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


def _score_sarimax_candidates(
    candidate_models: dict,
    prechampion_configs: dict,
    training_metadata: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_train_df: pd.DataFrame,
    params: dict,
    mase_period: int,
    require_all: bool,
) -> list[dict]:
    """Refit each SARIMAX prechampion on full_train, then forecast the test period."""
    date_col: str = params.get("date_column", "month_start_date")
    target_col: str = params.get("target_column", "monthly_demand")

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

    exog_cols: list[str] = list(training_metadata.get("exogenous_columns", []))

    full_train_df = full_train_df.copy()
    full_train_df[date_col] = pd.to_datetime(full_train_df[date_col])
    full_train_df = full_train_df.sort_values(date_col).reset_index(drop=True)

    test_df = test_df.copy()
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    test_df = test_df.sort_values(date_col).reset_index(drop=True)

    y_train = train_df[target_col].values.astype(float)
    y_true = test_df[target_col].values.astype(float)
    full_train_y = full_train_df[target_col].values.astype(float)
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
        config = entry.get("config", {})
        use_exog: bool = bool(config.get("use_exog", False))
        available_exog = [c for c in exog_cols if c in full_train_df.columns]

        try:
            full_train_exog = (
                full_train_df[available_exog].values.astype(float)
                if (use_exog and available_exog)
                else None
            )
            test_exog = (
                test_df[available_exog].values.astype(float)
                if (use_exog and available_exog)
                else None
            )

            model_obj = SARIMAX(
                endog=full_train_y,
                exog=full_train_exog,
                order=tuple(config["order"]),
                seasonal_order=tuple(config["seasonal_order"]),
                trend=config.get("trend"),
                enforce_stationarity=bool(config.get("enforce_stationarity", False)),
                enforce_invertibility=bool(config.get("enforce_invertibility", False)),
            )
            result = model_obj.fit(disp=False)

            forecast_out = result.get_forecast(steps=n_test, exog=test_exog)
            y_pred = np.asarray(forecast_out.predicted_mean, dtype=float)[:n_test]

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
                )
            )
            logger.info(
                "SARIMAX candidate '%s' scored — wape=%.4f",
                trial_id,
                rows[-1]["wape"] or float("nan"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SARIMAX candidate '%s' scoring failed: %s", trial_id, exc)

    if not rows and require_all:
        raise RuntimeError(
            "All SARIMAX prechampion candidates failed during test scoring."
        )
    return rows


def _build_metrics_row(
    family: str,
    candidate_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    mase_period: int,
    test_start: str,
    test_end: str,
    candidate_rank: int | None,
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

    return {
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
        "primary_metric": "wape",
        "primary_metric_value": wape_val,
        "is_family_champion": False,
        "is_production_champion": False,
    }


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
