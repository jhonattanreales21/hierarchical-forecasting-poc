"""Monthly Prophet model-selection nodes.

Evaluates pre-champion candidates on the held-out test set, computes
rolling-origin M-2/M-3 operational metrics, selects the final champion
based on configurable metrics (primary: WAPE), optionally refits on all
available historical data, and persists the champion model and metadata
for Stage 6 forecast inference.
"""

import logging
import warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from shared.metrics import mase as _shared_mase
from shared.metrics import wape as _shared_wape

# Suppress verbose Stan/cmdstanpy progress output during Prophet fitting
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Metrics where lower value = better rank (ascending sort).
_ERROR_METRICS: frozenset[str] = frozenset(
    {
        "wape",
        "mase",
        "mae",
        "rmse",
        "mape",
        "test_m2_wape",
        "test_m3_wape",
        "horizon_2_mae",
        "horizon_2_mape",
        "horizon_2_wmape",
        "horizon_3_mae",
        "horizon_3_mape",
        "horizon_3_wmape",
        # validation_rank is treated as ascending: rank 1 = best validation performance
        "validation_rank",
    }
)


# ── Public node functions ─────────────────────────────────────────────────────


def evaluate_monthly_prophet_prechampions_on_test(  # noqa: PLR0915
    monthly_prophet_test: pd.DataFrame,
    monthly_prophet_candidate_models: dict,
    monthly_prophet_prechampion_configs: dict,
    monthly_prophet_tuning_results: pd.DataFrame,
    monthly_prophet_train: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each pre-champion Prophet model on the held-out test set.

    Loads each pre-champion model, forecasts the test period, and computes full
    test metrics including WAPE (primary), MASE, and RMSE. MASE uses the training
    series for its naive denominator to prevent data leakage.

    Args:
        monthly_prophet_test: Test split with ds, y, sku, and active regressor columns.
        monthly_prophet_candidate_models: Dict mapping candidate_id → fitted Prophet model.
        monthly_prophet_prechampion_configs: Pre-champion configurations from Stage 4.
        monthly_prophet_tuning_results: Ranked tuning results from Stage 4 (unused here,
            included for pipeline lineage traceability).
        monthly_prophet_train: Training split used for the MASE naive denominator.
        params: Contents of ``model_selection.monthly_prophet`` from the parameter file.

    Returns:
        Two-element tuple:

        1. ``test_metrics`` — DataFrame, one row per pre-champion candidate with all
           test error metrics and status.
        2. ``test_forecast`` — DataFrame with stacked test-period forecasts for every
           successfully evaluated candidate (useful for diagnostic plots).

    Raises:
        ValueError: For empty test data, missing columns, or empty pre-champion list.
        RuntimeError: When all pre-champion candidates fail during test evaluation.
    """
    date_col: str = params.get("date_column", "ds")
    target_col: str = params.get("target_column", "y")
    sku_col: str = params.get("sku_column", "sku")

    metrics_cfg: dict = params.get("metrics", {})
    epsilon: float = float(metrics_cfg.get("epsilon", 1.0))
    mase_seasonal_period: int = int(metrics_cfg.get("mase_seasonal_period", 12))
    precision_threshold: float = float(
        params.get("selection", {}).get("business_success_precision_threshold", 0.85)
    )
    horizon_cfg: dict = metrics_cfg.get("horizon_metrics", {})
    horizons: list[int] = (
        [int(h) for h in horizon_cfg.get("horizons", [2, 3])]
        if horizon_cfg.get("enabled", True)
        else []
    )

    prechampions: list[dict] = monthly_prophet_prechampion_configs.get(
        "prechampions", []
    )
    if not prechampions:
        raise ValueError(
            "No pre-champion configurations found in monthly_prophet_prechampion_configs."
        )

    # Active regressors are shared across all pre-champions; read from the first entry
    active_regressors: list[str] = list(prechampions[0].get("active_regressors", []))

    _validate_test_inputs(monthly_prophet_test, active_regressors, date_col, target_col)

    test_df = monthly_prophet_test.copy()
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    test_df = test_df.sort_values(date_col).reset_index(drop=True)

    # y_train for MASE — use training split values (leakage-safe: no val/test data)
    train_df = monthly_prophet_train.copy()
    train_df[date_col] = pd.to_datetime(train_df[date_col])
    y_train_for_mase = train_df[target_col].values.astype(float)

    logger.info(
        "Test set: %d rows | %s → %s",
        len(test_df),
        test_df[date_col].min().date(),
        test_df[date_col].max().date(),
    )
    logger.info(
        "Train set for MASE: %d rows | seasonality=%d",
        len(y_train_for_mase),
        mase_seasonal_period,
    )
    logger.info("Pre-champions to evaluate: %d", len(prechampions))

    test_start = str(test_df[date_col].min().date())
    test_end = str(test_df[date_col].max().date())
    sku_val = test_df[sku_col].iloc[0] if sku_col in test_df.columns else None

    metrics_rows: list[dict] = []
    forecast_frames: list[pd.DataFrame] = []

    for prechampion in prechampions:
        candidate_id: str = str(prechampion["candidate_id"])

        if candidate_id not in monthly_prophet_candidate_models:
            msg = f"Model artifact not found for candidate {candidate_id}."
            logger.error(msg)
            metrics_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "error_message": msg,
                    "test_start_date": test_start,
                    "test_end_date": test_end,
                    "test_rows": len(test_df),
                }
            )
            continue

        try:
            model: Prophet = monthly_prophet_candidate_models[candidate_id]
            forecast = _forecast_test_period(
                model, test_df, active_regressors, date_col
            )

            y_true = test_df[target_col].values.astype(float)
            y_pred = forecast["yhat"].values.astype(float)

            base = _compute_forecast_metrics(
                y_true,
                y_pred,
                epsilon,
                precision_threshold,
                y_train=y_train_for_mase,
                mase_seasonal_period=mase_seasonal_period,
            )
            horiz = _compute_horizon_metrics(
                test_df,
                y_pred,
                date_col,
                target_col,
                horizons,
                epsilon,
                precision_threshold,
            )

            logger.info(
                "%s → wape=%.4f  mase=%s  rmse=%.2f  mape=%.4f",
                candidate_id,
                base["wape"] if base["wape"] is not None else float("nan"),
                f"{base['mase']:.4f}" if base["mase"] is not None else "NaN",
                base["rmse"],
                base["mape"],
            )

            metrics_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "success",
                    "error_message": None,
                    **base,
                    **horiz,
                    "test_start_date": test_start,
                    "test_end_date": test_end,
                    "test_rows": len(test_df),
                }
            )

            yhat_lower = (
                forecast["yhat_lower"].values
                if "yhat_lower" in forecast.columns
                else [None] * len(forecast)
            )
            yhat_upper = (
                forecast["yhat_upper"].values
                if "yhat_upper" in forecast.columns
                else [None] * len(forecast)
            )
            forecast_frames.append(
                pd.DataFrame(
                    {
                        "ds": test_df[date_col].values,
                        "sku": sku_val,
                        "y": y_true,
                        "yhat": y_pred,
                        "yhat_lower": yhat_lower,
                        "yhat_upper": yhat_upper,
                        "candidate_id": candidate_id,
                        "is_champion": False,  # updated downstream after selection
                        "dataset": "test",
                    }
                )
            )

        except Exception as exc:  # noqa: BLE001 — isolate per-candidate failures
            logger.error(
                "Candidate %s failed during test evaluation: %s | config=%s",
                candidate_id,
                exc,
                prechampion.get("model_params", {}),
            )
            metrics_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "error_message": str(exc),
                    "test_start_date": test_start,
                    "test_end_date": test_end,
                    "test_rows": len(test_df),
                }
            )

    n_success = sum(1 for r in metrics_rows if r["status"] == "success")
    n_failed = len(metrics_rows) - n_success
    logger.info(
        "Test evaluation complete — %d successful, %d failed.",
        n_success,
        n_failed,
    )

    if n_success == 0:
        raise RuntimeError(
            "All pre-champion candidates failed during test evaluation. "
            "Check logs for details."
        )

    test_metrics_df = pd.DataFrame(metrics_rows)
    test_forecast_df = (
        pd.concat(forecast_frames, ignore_index=True)
        if forecast_frames
        else pd.DataFrame()
    )

    logger.info(
        "Outputs — test_metrics=%s  test_forecast=%s",
        test_metrics_df.shape,
        test_forecast_df.shape,
    )
    return test_metrics_df, test_forecast_df


def evaluate_monthly_prophet_rolling_origin_metrics(  # noqa: PLR0912, PLR0915
    monthly_prophet_full_train: pd.DataFrame,
    monthly_prophet_test: pd.DataFrame,
    monthly_prophet_prechampion_configs: dict,
    monthly_prophet_split_metadata: dict,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling-origin M-2 and M-3 lead-time metrics over the test window.

    For each pre-champion and each valid origin-target pair, refits the model on
    history up to the origin month and evaluates the forecast at the target month
    (2 or 3 months ahead). Only target months that fall within the test window are
    evaluated. No information after the origin month is used.

    Args:
        monthly_prophet_full_train: Full historical dataset (train + val + test rows)
            used to slice history up to each operational origin.
        monthly_prophet_test: Test split with actual demand for the target months.
        monthly_prophet_prechampion_configs: Pre-champion configurations with
            fixed hyperparameters from Stage 4.
        monthly_prophet_split_metadata: Split metadata providing train and test date ranges.
        params: Contents of ``model_selection.monthly_prophet`` from the parameter file.

    Returns:
        Two-element tuple:

        1. ``operational_test_forecasts`` — DataFrame with one row per
           (candidate, origin, target, lead_time) combination.
        2. ``operational_lead_time_metrics`` — DataFrame with aggregated WAPE per
           candidate and lead time (test_m2_wape, test_m3_wape).
    """
    date_col: str = params.get("date_column", "ds")
    target_col: str = params.get("target_column", "y")
    regressor_mode: str = params.get("regressors", {}).get("mode", "additive")

    op_cfg: dict = params.get("operational_lead_time", {})
    enabled: bool = bool(op_cfg.get("enabled", True))
    lead_times: list[int] = [int(lt) for lt in op_cfg.get("lead_times", [2, 3])]

    prechampions: list[dict] = monthly_prophet_prechampion_configs.get(
        "prechampions", []
    )
    active_regressors: list[str] = (
        list(prechampions[0].get("active_regressors", [])) if prechampions else []
    )

    if not enabled or not prechampions:
        logger.info(
            "Rolling-origin evaluation is disabled or no pre-champions available."
        )
        return pd.DataFrame(), pd.DataFrame()

    # Determine test window from split metadata
    test_meta = monthly_prophet_split_metadata.get("test", {})
    test_start = pd.Timestamp(test_meta.get("start_date"))
    test_end = pd.Timestamp(test_meta.get("end_date"))

    # Earliest operational origin = 1 month before test_start
    earliest_origin = test_start - pd.DateOffset(months=1)

    # Enumerate all valid (origin, target, lead_time) triples
    # origin iterates from earliest_origin; only include if target falls in test window
    valid_pairs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for lt in lead_times:
        origin = earliest_origin
        while True:
            target = origin + pd.DateOffset(months=lt)
            target = target.normalize()
            if target > test_end:
                break
            if target >= test_start:
                valid_pairs.append((origin.normalize(), target, lt))
            origin = origin + pd.DateOffset(months=1)

    if not valid_pairs:
        logger.warning(
            "No valid rolling-origin pairs found for lead times %s in test window %s → %s.",
            lead_times,
            test_start.date(),
            test_end.date(),
        )
        return pd.DataFrame(), pd.DataFrame()

    logger.info(
        "Rolling-origin evaluation: %d valid pairs over test window %s → %s",
        len(valid_pairs),
        test_start.date(),
        test_end.date(),
    )

    # Prepare full history and test actuals
    full_df = monthly_prophet_full_train.copy()
    full_df[date_col] = pd.to_datetime(full_df[date_col])
    full_df = full_df.sort_values(date_col).reset_index(drop=True)

    test_df = monthly_prophet_test.copy()
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    # Merge test actuals into a lookup keyed by date
    actuals_lookup = dict(zip(test_df[date_col].dt.normalize(), test_df[target_col]))

    forecast_rows: list[dict] = []

    for prechampion in prechampions:
        candidate_id = str(prechampion["candidate_id"])
        model_params = dict(prechampion.get("model_params", {}))

        for origin, target, lt in valid_pairs:
            # Slice history up to and including origin (leakage-safe)
            history = full_df[full_df[date_col].dt.normalize() <= origin].copy()
            if history.empty:
                logger.warning(
                    "No history available for origin %s; skipping pair (%s, %s, L=%d).",
                    origin.date(),
                    origin.date(),
                    target.date(),
                    lt,
                )
                continue

            training_cutoff = history[date_col].max().normalize()

            # Build regressor row for the target month from test_df
            target_rows = test_df[test_df[date_col].dt.normalize() == target]
            if target_rows.empty:
                logger.warning(
                    "No regressor data available for target month %s; skipping.",
                    target.date(),
                )
                continue

            y_true_val = actuals_lookup.get(target)
            if y_true_val is None:
                continue

            try:
                model = _refit_prophet_on_history(
                    history,
                    model_params,
                    active_regressors,
                    date_col,
                    target_col,
                    regressor_mode,
                )
                # Forecast from origin for (lead_time + 1) periods to reach target
                predict_input = target_rows[[date_col] + active_regressors].rename(
                    columns={date_col: "ds"}
                )
                fc = model.predict(predict_input)
                y_pred_val = float(fc["yhat"].iloc[0])

                forecast_rows.append(
                    {
                        "model_family": "prophet",
                        "candidate_id": candidate_id,
                        "split": "test",
                        "origin_month": origin,
                        "target_month": target,
                        "lead_time": lt,
                        "training_cutoff": training_cutoff,
                        "y_true": float(y_true_val),
                        "y_pred": y_pred_val,
                    }
                )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Rolling-origin refit failed for %s origin=%s target=%s: %s",
                    candidate_id,
                    origin.date(),
                    target.date(),
                    exc,
                )

    if not forecast_rows:
        logger.warning("Rolling-origin evaluation produced no forecast rows.")
        return pd.DataFrame(), pd.DataFrame()

    raw_df = pd.DataFrame(forecast_rows)

    # Aggregate WAPE per candidate per lead time
    agg_rows: list[dict] = []
    for candidate_id in raw_df["candidate_id"].unique():
        row: dict = {"candidate_id": candidate_id}
        for lt in lead_times:
            col_name = f"test_m{lt}_wape"
            pairs_col = f"n_m{lt}_pairs"
            subset = raw_df[
                (raw_df["candidate_id"] == candidate_id) & (raw_df["lead_time"] == lt)
            ]
            n_pairs = len(subset)
            row[pairs_col] = n_pairs
            if n_pairs == 0:
                warnings.warn(
                    f"No valid M-{lt} pairs for candidate {candidate_id}. "
                    f"Setting test_m{lt}_wape to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                row[col_name] = float("nan")
            else:
                row[col_name] = _shared_wape(
                    subset["y_true"].values, subset["y_pred"].values
                )
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    logger.info(
        "Rolling-origin complete — %d forecast rows, %d candidates aggregated.",
        len(raw_df),
        len(agg_df),
    )
    return raw_df, agg_df


def select_monthly_prophet_champion(
    monthly_prophet_test_metrics: pd.DataFrame,
    monthly_prophet_tuning_results: pd.DataFrame,
    monthly_prophet_prechampion_configs: dict,
    monthly_prophet_split_metadata: dict,
    monthly_operational_lead_time_metrics: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Rank pre-champions on test metrics and select one final champion.

    Ranks candidates by WAPE (primary), then available operational lead-time
    metrics (M-3 WAPE, M-2 WAPE), then MASE and RMSE. Missing operational
    metrics are skipped per the configured policy.

    Args:
        monthly_prophet_test_metrics: Output of evaluate_monthly_prophet_prechampions_on_test.
        monthly_prophet_tuning_results: Stage 4 ranked results (provides validation rank).
        monthly_prophet_prechampion_configs: Pre-champion configurations from Stage 4.
        monthly_prophet_split_metadata: Split window metadata from model_input_preparation.
        monthly_operational_lead_time_metrics: Aggregated rolling-origin metrics
            (test_m2_wape, test_m3_wape, n_m2_pairs, n_m3_pairs) per candidate.
        params: Contents of ``model_selection.monthly_prophet`` from the parameter file.

    Returns:
        Three-element tuple:

        1. ``selection_summary`` — DataFrame with one row per candidate.
        2. ``champion_metadata`` — JSON-serialisable dict for Stage 6 inference.
        3. ``model_selection_audit`` — DataFrame with selection reasoning and metric notes.

    Raises:
        ValueError: If no successful candidates are available for selection.
        RuntimeError: If the champion config is missing from prechampion_configs.
    """
    selection_cfg: dict = params.get("selection", {})
    primary_metric: str = selection_cfg.get("primary_metric", "wape")
    tie_breakers: list[str] = list(
        selection_cfg.get(
            "tie_breakers",
            ["test_m3_wape", "test_m2_wape", "mase", "rmse", "validation_rank"],
        )
    )
    missing_policy: str = selection_cfg.get(
        "missing_metric_policy", "skip_with_audit_note"
    )
    precision_threshold: float = float(
        selection_cfg.get("business_success_precision_threshold", 0.85)
    )
    refit_enabled: bool = bool(params.get("refit_champion", {}).get("enabled", True))

    prechampions: list[dict] = monthly_prophet_prechampion_configs.get(
        "prechampions", []
    )

    success_mask = monthly_prophet_test_metrics["status"] == "success"
    success_df = monthly_prophet_test_metrics[success_mask].copy()
    failed_df = monthly_prophet_test_metrics[~success_mask].copy()

    if success_df.empty:
        raise ValueError(
            "No successful pre-champion candidates available for champion selection."
        )

    # Merge validation rank for tie-breaking
    val_ranks = monthly_prophet_tuning_results[["candidate_id", "rank"]].rename(
        columns={"rank": "validation_rank"}
    )
    success_df = success_df.merge(val_ranks, on="candidate_id", how="left")

    # Merge operational lead-time metrics when available
    missing_op_metrics: list[str] = []
    if not monthly_operational_lead_time_metrics.empty:
        op_cols = [
            c
            for c in monthly_operational_lead_time_metrics.columns
            if c != "candidate_id"
        ]
        success_df = success_df.merge(
            monthly_operational_lead_time_metrics[["candidate_id"] + op_cols],
            on="candidate_id",
            how="left",
        )
        for col in ["test_m2_wape", "test_m3_wape"]:
            if col in success_df.columns and success_df[col].isna().all():
                missing_op_metrics.append(col)
                logger.warning(
                    "Operational metric %r is NaN for all candidates. "
                    "Excluding from sort order per policy=%r.",
                    col,
                    missing_policy,
                )
    else:
        logger.warning(
            "monthly_operational_lead_time_metrics is empty. "
            "Operational metrics will not contribute to champion selection."
        )
        missing_op_metrics = ["test_m2_wape", "test_m3_wape"]

    # Remove missing operational metrics from tie-breakers
    active_tie_breakers = [tb for tb in tie_breakers if tb not in missing_op_metrics]

    ranked_df = _rank_test_candidates(success_df, primary_metric, active_tie_breakers)
    champion_row = ranked_df[ranked_df["test_rank"] == 1].iloc[0]
    champion_id: str = str(champion_row["candidate_id"])

    primary_val = champion_row.get(primary_metric)
    logger.info(
        "Champion selected: %s | %s=%.4f",
        champion_id,
        primary_metric,
        float(primary_val) if primary_val is not None else float("nan"),
    )

    fp = champion_row.get("forecast_precision")
    if fp is not None and float(fp) >= precision_threshold:
        logger.info(
            "Champion meets business success threshold (≥%.0f%% precision): %.4f",
            precision_threshold * 100,
            float(fp),
        )
    else:
        logger.warning(
            "Champion does NOT exceed the %.0f%% precision threshold. "
            "Best precision: %.4f",
            precision_threshold * 100,
            float(fp) if fp is not None else 0.0,
        )

    # Mark champion and build selection_reason
    ranked_df["is_champion"] = ranked_df["candidate_id"] == champion_id
    ranked_df["primary_metric"] = primary_metric
    ranked_df["primary_metric_value"] = ranked_df.get(primary_metric)
    ranked_df["selection_reason"] = ranked_df.apply(
        lambda r: _build_selection_reason(
            r, champion_id, primary_metric, active_tie_breakers, missing_op_metrics
        ),
        axis=1,
    )

    # Append failed candidates so the summary covers all pre-champions
    failed_df["test_rank"] = None
    failed_df["validation_rank"] = None
    failed_df["is_champion"] = False
    failed_df["primary_metric"] = primary_metric
    failed_df["primary_metric_value"] = None
    failed_df["selection_reason"] = ""

    summary_cols = [
        "candidate_id",
        "validation_rank",
        "test_rank",
        "is_champion",
        "primary_metric",
        "primary_metric_value",
        "wape",
        "mase",
        "mae",
        "rmse",
        "mape",
        "wmape",
        "test_m2_wape",
        "test_m3_wape",
        "forecast_precision",
        "business_success_flag",
        "selection_reason",
    ]
    ranked_cols = [c for c in summary_cols if c in ranked_df.columns]
    failed_cols = [c for c in summary_cols if c in failed_df.columns]
    selection_summary_df = pd.concat(
        [ranked_df[ranked_cols], failed_df[failed_cols]],
        ignore_index=True,
        sort=False,
    )

    # ── Build champion metadata ───────────────────────────────────────────────
    champion_config = next(
        (p for p in prechampions if str(p["candidate_id"]) == champion_id),
        None,
    )
    if champion_config is None:
        raise RuntimeError(
            f"Champion config not found for {champion_id!r} in prechampion_configs."
        )

    val_metrics = champion_config.get("validation_metrics", {})
    test_row = (
        monthly_prophet_test_metrics[
            monthly_prophet_test_metrics["candidate_id"] == champion_id
        ]
        .iloc[0]
        .to_dict()
    )
    metric_keys = ["wape", "mase", "mae", "rmse", "mape", "wmape", "forecast_precision"]

    train_summary = monthly_prophet_split_metadata.get("full_train", {})
    test_summary = monthly_prophet_split_metadata.get("test", {})

    champion_trial_row = monthly_prophet_tuning_results[
        monthly_prophet_tuning_results["candidate_id"] == champion_id
    ]
    best_trial_number = (
        _safe_int(champion_trial_row["trial_number"].iloc[0])
        if not champion_trial_row.empty and "trial_number" in champion_trial_row.columns
        else None
    )

    # Operational metrics for champion
    op_test_m2_wape = None
    op_test_m3_wape = None
    n_m2_pairs = None
    n_m3_pairs = None
    if not monthly_operational_lead_time_metrics.empty:
        champ_op = monthly_operational_lead_time_metrics[
            monthly_operational_lead_time_metrics["candidate_id"] == champion_id
        ]
        if not champ_op.empty:
            op_test_m2_wape = _safe_float(
                champ_op.get("test_m2_wape", pd.Series([None])).iloc[0]
            )
            op_test_m3_wape = _safe_float(
                champ_op.get("test_m3_wape", pd.Series([None])).iloc[0]
            )
            n_m2_pairs = _safe_int(
                champ_op.get("n_m2_pairs", pd.Series([None])).iloc[0]
            )
            n_m3_pairs = _safe_int(
                champ_op.get("n_m3_pairs", pd.Series([None])).iloc[0]
            )

    champion_metadata: dict = {
        "model_family": "prophet",
        "granularity": "monthly",
        "champion_id": champion_id,
        "selection_stage": "model_selection",
        "selection_metric": primary_metric,
        "selection_metric_value": _safe_float(champion_row.get(primary_metric)),
        "business_success_precision_threshold": precision_threshold,
        "business_success_flag": bool(champion_row.get("business_success_flag", False)),
        "refit_on_full_train": refit_enabled,
        "selected_at": datetime.now(tz=UTC).isoformat(),
        "active_regressors": list(champion_config.get("active_regressors", [])),
        "model_params": dict(champion_config.get("model_params", {})),
        "validation_metrics": {k: _safe_float(val_metrics.get(k)) for k in metric_keys},
        "test_metrics": {k: _safe_float(test_row.get(k)) for k in metric_keys},
        "operational_test_metrics": {
            "test_m2_wape": op_test_m2_wape,
            "test_m3_wape": op_test_m3_wape,
            "n_m2_pairs": n_m2_pairs,
            "n_m3_pairs": n_m3_pairs,
        },
        "optuna_best_trial": {
            "trial_number": best_trial_number,
            "training_metadata_artifact": "monthly_prophet_training_metadata",
        },
        "train_window": {
            "start_date": train_summary.get("start_date"),
            "end_date": train_summary.get("end_date"),
        },
        "test_window": {
            "start_date": test_summary.get("start_date"),
            "end_date": test_summary.get("end_date"),
        },
    }

    # ── Build model selection audit ───────────────────────────────────────────
    audit_rows: list[dict] = []
    for _, row in selection_summary_df.iterrows():
        cid = str(row.get("candidate_id", ""))
        audit_rows.append(
            {
                "candidate_id": cid,
                "is_champion": bool(row.get("is_champion", False)),
                "test_rank": row.get("test_rank"),
                "validation_rank": row.get("validation_rank"),
                "primary_metric": primary_metric,
                "wape": _safe_float(row.get("wape")),
                "mase": _safe_float(row.get("mase")),
                "rmse": _safe_float(row.get("rmse")),
                "mape": _safe_float(row.get("mape")),
                "test_m2_wape": _safe_float(row.get("test_m2_wape")),
                "test_m3_wape": _safe_float(row.get("test_m3_wape")),
                "n_m2_pairs": row.get("n_m2_pairs"),
                "n_m3_pairs": row.get("n_m3_pairs"),
                "missing_operational_metrics": ", ".join(missing_op_metrics) or None,
                "selection_reason": str(row.get("selection_reason", "")),
                "audited_at": datetime.now(tz=UTC).isoformat(),
            }
        )
    audit_df = pd.DataFrame(audit_rows)

    logger.info(
        "Champion metadata built — id=%s  test_%s=%.4f  refit=%s  "
        "test_m2_wape=%s  test_m3_wape=%s",
        champion_id,
        primary_metric,
        _safe_float(champion_row.get(primary_metric)) or 0.0,
        refit_enabled,
        f"{op_test_m2_wape:.4f}" if op_test_m2_wape else "N/A",
        f"{op_test_m3_wape:.4f}" if op_test_m3_wape else "N/A",
    )
    return selection_summary_df, champion_metadata, audit_df


def build_monthly_prophet_champion_model(
    monthly_prophet_full_train: pd.DataFrame,
    monthly_prophet_candidate_models: dict,
    monthly_prophet_champion_metadata: dict,
    params: dict,
) -> Any:
    """Retrieve or refit the final champion model for downstream inference.

    Champion selection and test metrics are computed from the pre-champion model
    as fitted during Stage 4.  This node runs AFTER selection and optionally
    refits the champion configuration on all available historical data so that
    inference benefits from the full training window.

    Reported test metrics are always based on the pre-refit model and are not
    affected by the refit performed here.

    Args:
        monthly_prophet_full_train: Combined train + validation dataset.
        monthly_prophet_candidate_models: Dict mapping candidate_id → fitted Prophet.
        monthly_prophet_champion_metadata: Champion metadata from select_monthly_prophet_champion.
        params: Contents of ``model_selection.monthly_prophet`` from the parameter file.

    Returns:
        Fitted Prophet model — either the original pre-champion model or a model
        refitted on monthly_prophet_full_train, depending on refit_champion.enabled.

    Raises:
        RuntimeError: When refit is disabled and the pre-champion model is not found.
    """
    refit_cfg: dict = params.get("refit_champion", {})
    refit_enabled: bool = bool(refit_cfg.get("enabled", True))

    champion_id: str = monthly_prophet_champion_metadata["champion_id"]
    active_regressors: list[str] = list(
        monthly_prophet_champion_metadata.get("active_regressors", [])
    )
    model_params: dict = dict(monthly_prophet_champion_metadata.get("model_params", {}))

    if not refit_enabled:
        logger.info(
            "Champion refit disabled — returning pre-champion model %s as-is.",
            champion_id,
        )
        if champion_id not in monthly_prophet_candidate_models:
            raise RuntimeError(
                f"Pre-champion model {champion_id!r} not found in candidate_models artifact."
            )
        return monthly_prophet_candidate_models[champion_id]

    date_col: str = params.get("date_column", "ds")
    target_col: str = params.get("target_column", "y")
    regressor_mode: str = params.get("regressors", {}).get("mode", "additive")

    logger.info(
        "Refitting champion %s on full training data (%d rows) …",
        champion_id,
        len(monthly_prophet_full_train),
    )

    champion_model = _refit_prophet_champion(
        monthly_prophet_full_train,
        model_params,
        active_regressors,
        date_col,
        target_col,
        regressor_mode,
    )

    logger.info("Champion model successfully refitted on full historical data.")
    return champion_model


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_test_inputs(
    test_df: pd.DataFrame,
    active_regressors: list[str],
    date_col: str,
    target_col: str,
) -> None:
    """Raise a descriptive error for any invalid test input."""
    if test_df.empty:
        raise ValueError("Test DataFrame is empty.")

    required_cols = [date_col, target_col] + active_regressors
    missing = [c for c in required_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Required columns missing from test data: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(test_df[date_col]):
        try:
            pd.to_datetime(test_df[date_col])
        except Exception as exc:
            raise ValueError(
                f"Column {date_col!r} could not be parsed as datetime."
            ) from exc

    if not pd.api.types.is_numeric_dtype(test_df[target_col]):
        raise ValueError(
            f"Target column {target_col!r} must be numeric; "
            f"found dtype {test_df[target_col].dtype}."
        )

    for col in active_regressors:
        n_null = int(test_df[col].isnull().sum())
        if n_null > 0:
            raise ValueError(
                f"Active regressor {col!r} has {n_null} null value(s) in test data."
            )


def _forecast_test_period(
    model: Prophet,
    test_df: pd.DataFrame,
    active_regressors: list[str],
    date_col: str,
) -> pd.DataFrame:
    """Generate test-period forecasts from a fitted Prophet model."""
    predict_input = test_df[[date_col] + active_regressors].rename(
        columns={date_col: "ds"}
    )
    return model.predict(predict_input)


def _compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float,
    precision_threshold: float,
    y_train: np.ndarray | None = None,
    mase_seasonal_period: int = 12,
) -> dict:
    """Compute scalar forecast accuracy metrics for a single candidate.

    WAPE (primary) uses the clean sum-based formula; returns NaN when denominator
    is zero. MASE uses the training series for the naive denominator (no leakage).
    MAPE is retained as a secondary diagnostic with an epsilon guard.

    Args:
        y_true: Observed values.
        y_pred: Forecasted values, same shape as y_true.
        epsilon: Small constant added to |y_true| in the epsilon-guarded MAPE denominator.
        precision_threshold: Threshold above which business_success_flag is True.
        y_train: Training series for the MASE naive denominator. NaN when None.
        mase_seasonal_period: Seasonal period for the MASE naive benchmark.

    Returns:
        Dict with wape, mase, mae, rmse, mape, wmape, forecast_precision, business_success_flag.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_errors = np.abs(y_true - y_pred)

    mae = float(np.mean(abs_errors))
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape_val = float(np.mean(abs_errors / (np.abs(y_true) + epsilon)))
    wmape = float(np.sum(abs_errors) / (np.sum(np.abs(y_true)) + epsilon))
    wape_val = _shared_wape(y_true, y_pred)
    forecast_precision = 1.0 - mape_val
    business_success_flag = bool(forecast_precision >= precision_threshold)

    mase_val: float | None = None
    if y_train is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            raw_mase = _shared_mase(y_true, y_pred, y_train, mase_seasonal_period)
        mase_val = raw_mase if np.isfinite(raw_mase) else None

    return {
        "wape": wape_val,
        "mase": mase_val,
        "mae": mae,
        "rmse": rmse_val,
        "mape": mape_val,
        "wmape": wmape,
        "forecast_precision": forecast_precision,
        "business_success_flag": business_success_flag,
    }


def _compute_horizon_metrics(  # noqa: PLR0913
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    date_col: str,
    target_col: str,
    horizons: list[int],
    epsilon: float,
    precision_threshold: float,
) -> dict:
    """Compute point metrics at specified forecast horizons."""
    test_sorted = test_df.sort_values(date_col).reset_index(drop=True)
    y_true_all = test_sorted[target_col].values.astype(float)
    result: dict = {}

    for h in horizons:
        idx = h - 1  # convert 1-based horizon to 0-based index
        if idx >= len(test_sorted):
            logger.warning(
                "Test set has %d rows; horizon-%d metrics unavailable.",
                len(test_sorted),
                h,
            )
            result[f"horizon_{h}_mae"] = None
            result[f"horizon_{h}_mape"] = None
            result[f"horizon_{h}_wmape"] = None
            result[f"horizon_{h}_forecast_precision"] = None
        else:
            pt = _compute_forecast_metrics(
                np.array([y_true_all[idx]]),
                np.array([float(y_pred[idx])]),
                epsilon,
                precision_threshold,
            )
            result[f"horizon_{h}_mae"] = pt["mae"]
            result[f"horizon_{h}_mape"] = pt["mape"]
            result[f"horizon_{h}_wmape"] = pt["wmape"]
            result[f"horizon_{h}_forecast_precision"] = pt["forecast_precision"]

    return result


def _rank_test_candidates(
    success_df: pd.DataFrame,
    primary_metric: str,
    tie_breakers: list[str],
) -> pd.DataFrame:
    """Rank candidates by primary metric with tie-breaker fallback.

    Error metrics (listed in _ERROR_METRICS) are ranked ascending (lower is better).
    Precision-style metrics are ranked descending (higher is better).
    Columns absent from the DataFrame are silently skipped.
    """
    all_sort_cols = [primary_metric] + [
        tb for tb in tie_breakers if tb in success_df.columns
    ]
    ascending_flags = [col in _ERROR_METRICS for col in all_sort_cols]

    ranked = success_df.sort_values(
        all_sort_cols,
        ascending=ascending_flags,
        na_position="last",
    ).reset_index(drop=True)
    ranked["test_rank"] = list(range(1, len(ranked) + 1))
    return ranked


def _build_selection_reason(
    row: pd.Series,
    champion_id: str,
    primary_metric: str,
    tie_breakers: list[str],
    missing_op_metrics: list[str],
) -> str:
    """Build a short human-readable selection reason for the summary table."""
    if str(row["candidate_id"]) != champion_id:
        return ""
    metric_val = row.get(primary_metric)
    val_str = f"{float(metric_val):.4f}" if metric_val is not None else "N/A"
    tie_label = (
        ", ".join(tb.upper() for tb in tie_breakers[:2]) if tie_breakers else "none"
    )
    missing_note = (
        f" (missing operational metrics excluded from ranking: {', '.join(missing_op_metrics)})"
        if missing_op_metrics
        else ""
    )
    return (
        f"Selected because it achieved the lowest test {primary_metric.upper()} "
        f"({val_str}) among pre-champion candidates, with {tie_label} as tie-breaker"
        f"{missing_note}."
    )


def _refit_prophet_on_history(  # noqa: PLR0913
    history_df: pd.DataFrame,
    model_params: dict,
    active_regressors: list[str],
    date_col: str,
    target_col: str,
    regressor_mode: str,
) -> Prophet:
    """Refit a Prophet model on a history slice for rolling-origin evaluation."""
    fit_df = history_df.copy()
    fit_df[date_col] = pd.to_datetime(fit_df[date_col])
    fit_df = fit_df.sort_values(date_col).reset_index(drop=True)
    train_fit = fit_df[[date_col, target_col] + active_regressors].rename(
        columns={date_col: "ds", target_col: "y"}
    )
    return _build_and_fit_prophet(
        train_fit, model_params, active_regressors, regressor_mode
    )


def _refit_prophet_champion(  # noqa: PLR0913
    full_train_df: pd.DataFrame,
    model_params: dict,
    active_regressors: list[str],
    date_col: str,
    target_col: str,
    regressor_mode: str,
) -> Prophet:
    """Refit a Prophet model on the full historical training dataset."""
    fit_df = full_train_df.copy()
    fit_df[date_col] = pd.to_datetime(fit_df[date_col])
    fit_df = fit_df.sort_values(date_col).reset_index(drop=True)
    train_fit = fit_df[[date_col, target_col] + active_regressors].rename(
        columns={date_col: "ds", target_col: "y"}
    )
    return _build_and_fit_prophet(
        train_fit, model_params, active_regressors, regressor_mode
    )


def _build_and_fit_prophet(
    train_fit: pd.DataFrame,
    model_params: dict,
    active_regressors: list[str],
    regressor_mode: str,
) -> Prophet:
    """Instantiate, configure, and fit a Prophet model."""
    model = Prophet(
        changepoint_prior_scale=float(
            model_params.get("changepoint_prior_scale", 0.05)
        ),
        seasonality_prior_scale=float(
            model_params.get("seasonality_prior_scale", 10.0)
        ),
        holidays_prior_scale=float(model_params.get("holidays_prior_scale", 10.0)),
        seasonality_mode=str(model_params.get("seasonality_mode", "additive")),
        yearly_seasonality=bool(model_params.get("yearly_seasonality", True)),
        weekly_seasonality=bool(model_params.get("weekly_seasonality", False)),
        daily_seasonality=bool(model_params.get("daily_seasonality", False)),
        interval_width=float(model_params.get("interval_width", 0.8)),
    )
    for regressor_name in active_regressors:
        model.add_regressor(regressor_name, mode=regressor_mode)
    model.fit(train_fit)
    return model


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
    """Convert to Python int, returning None for missing values."""
    if value is None:
        return None
    try:
        f = float(value)
        return int(f) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
