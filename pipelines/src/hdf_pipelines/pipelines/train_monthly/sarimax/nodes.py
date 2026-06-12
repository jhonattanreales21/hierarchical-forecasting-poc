"""Monthly SARIMAX training nodes (rolling-origin protocol).

Each Optuna trial is scored by a rolling-origin backtest: for every cycle SARIMAX
is refit on history through the origin and forecasts the next ``horizon`` months;
WMAPE/BIAS metrics are pooled across cycles and the objective is ``WMAPE_M3``. A
Ljung-Box residual test on the last rolling-origin cycle fit marks each candidate's
eligibility; ineligible candidates are kept out of the pre-champion shortlist when
any eligible candidate exists.
"""

import logging
import time
import warnings as _warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from shared.rolling_origin import RollingOriginCycle, run_rolling_origin
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from hdf_pipelines.pipelines.train_monthly.nodes import (
    build_monthly_rolling_origin_cycles,
    extract_rolling_origin_metric_set,
    log_trial_predictions,
    make_pruning_callback,
)
from hdf_pipelines.utils.optuna_helpers import (
    build_rolling_origin_pruner,
    create_optuna_study,
    suggest_trial_params,
    validate_optuna_search_space,
)

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial verbose logging — SARIMAX already emits its own.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Public node ───────────────────────────────────────────────────────────────


def train_and_evaluate_monthly_sarimax_candidates(  # noqa: PLR0915
    monthly_sarimax_full_train: pd.DataFrame,
    monthly_sarimax_split_metadata: dict,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict]:
    """Tune SARIMAX with a rolling-origin backtest and emit candidate artifacts.

    Runs an Optuna TPE study over the SARIMAX search space. Each trial is scored by
    a rolling-origin backtest (objective: pooled ``WMAPE_M3``) and gets a
    Ljung-Box eligibility flag from the last cycle fit. The top-N eligible
    candidates are persisted (refit on full history) as pre-champions.

    Args:
        monthly_sarimax_full_train: SARIMAX-ready full history (full history) with
            date, target, and exogenous columns.
        monthly_sarimax_split_metadata: Metadata from the SARIMAX input adapter
            (column names, exogenous columns).
        params: Contents of ``train_monthly.sarimax`` from the parameter file.

    Returns:
        Six-element tuple: tuning_results, rolling_origin_metrics, prechampion_configs,
        candidate_models, training_metadata, candidate_monthly_sarimax (rank-1).

    Raises:
        ValueError: When inputs are invalid or required columns are missing.
        RuntimeError: When every configured trial fails to evaluate.
    """
    # ── extract configuration ─────────────────────────────────────────────────
    date_col: str = params.get("date_column", "month_start_date")
    target_col: str = params.get("target_column", "monthly_demand")
    objective_cfg: dict = params.get("objective", {})
    objective_metric: str = str(objective_cfg.get("metric", "wmape_m3"))
    objective_direction: str = str(objective_cfg.get("direction", "minimize")).lower()
    top_n: int = int(params.get("top_n_prechampions", 3))
    metrics_cfg: dict = params.get("metrics", {})
    mase_seasonal_period: int = int(metrics_cfg.get("mase_seasonal_period", 12))
    epsilon: float = float(metrics_cfg.get("epsilon", 1.0))
    tuning_cfg: dict = params.get("tuning", {})
    use_exog: bool = bool(tuning_cfg.get("use_exog", True))
    max_failed_trials: int | None = (
        int(params["max_failed_trials"])
        if params.get("max_failed_trials") is not None
        else None
    )

    ljung_cfg: dict = dict(params.get("ljung_box", {}))
    lb_enabled: bool = bool(ljung_cfg.get("enabled", True))
    lb_lags: int = int(ljung_cfg.get("lags", 10))
    lb_threshold: float = float(ljung_cfg.get("pvalue_threshold", 0.05))

    rolling_origin_cfg: dict = dict(tuning_cfg.get("rolling_origin", {}))
    pruning_cfg: dict = dict(tuning_cfg.get("pruning", {}))
    horizon: int = int(rolling_origin_cfg.get("horizon", 3))

    max_trials: int = int(tuning_cfg.get("max_trials", 40))
    sampler_cfg: dict = dict(tuning_cfg.get("sampler", {}))
    search_space: dict = validate_optuna_search_space(dict(tuning_cfg.get("search_space", {})))
    fixed_params: dict = dict(tuning_cfg.get("fixed_params", {}))
    s: int = int(fixed_params.get("s", mase_seasonal_period))

    # ── validate inputs ───────────────────────────────────────────────────────
    _validate_sarimax_training_inputs(monthly_sarimax_full_train, date_col, target_col)

    exog_cols: list[str] = list(monthly_sarimax_split_metadata.get("exogenous_columns", []))
    if use_exog and not exog_cols:
        logger.info(
            "use_exog=True but no exogenous columns in split metadata — running without exog."
        )
        use_exog = False

    full_df = monthly_sarimax_full_train.copy()
    full_df[date_col] = pd.to_datetime(full_df[date_col])
    full_df = full_df.sort_values(date_col).reset_index(drop=True)

    full_y = full_df[target_col].to_numpy(dtype=float)
    full_exog = _extract_exog(full_df, exog_cols) if use_exog else None

    cycles = build_monthly_rolling_origin_cycles(full_df, date_col, rolling_origin_cfg)
    logger.info(
        "SARIMAX training — %d rows | %s → %s | %d cycles (H=%d)  exog=%s  "
        "objective=%s  max_trials=%d  ljung_box=%s",
        len(full_df), full_df[date_col].min().date(), full_df[date_col].max().date(),
        len(cycles), horizon, exog_cols if use_exog else "(none)",
        objective_metric, max_trials, lb_enabled,
    )

    # ── Optuna study setup ────────────────────────────────────────────────────
    study = create_optuna_study(
        objective_direction, sampler_cfg, pruner=build_rolling_origin_pruner(pruning_cfg)
    )
    trial_results: list[dict] = []
    full_history_fits: dict[str, Any] = {}
    n_failed_count: list[int] = [0]

    def _stop_on_max_failures(
        _study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state == optuna.trial.TrialState.FAIL:
            n_failed_count[0] += 1
            if max_failed_trials is not None and n_failed_count[0] >= max_failed_trials:
                logger.warning(
                    "SARIMAX Optuna search: max_failed_trials=%d reached after %d "
                    "trial(s). Stopping early.",
                    max_failed_trials, trial.number + 1,
                )
                _study.stop()

    def objective(trial: optuna.Trial) -> float:
        trial_id = f"sarimax_trial_{trial.number + 1:03d}"
        t0 = time.perf_counter()
        raw_params = suggest_trial_params(trial, search_space, fixed_params)
        config = _build_sarimax_config_from_trial(raw_params, s)

        _trial_preds: list[dict] = []

        def _fit_forecast(train_df: pd.DataFrame, cycle: RollingOriginCycle) -> np.ndarray:
            y_pred = _sarimax_cycle_forecast(
                config, train_df, cycle, full_df, date_col, target_col, exog_cols, use_exog
            )
            _trial_preds.append({
                "target_start": cycle.target_dates[0].strftime("%Y-%m"),
                "target_end": cycle.target_dates[-1].strftime("%Y-%m"),
                "y_pred": y_pred.tolist(),
            })
            return y_pred

        on_cycle_end = make_pruning_callback(trial, pruning_cfg, metric_key=objective_metric)
        _, aggregated = run_rolling_origin(
            full_df, date_col, target_col, cycles, _fit_forecast,
            season=mase_seasonal_period, epsilon=epsilon, on_cycle_end=on_cycle_end,
        )
        metric_set = extract_rolling_origin_metric_set(aggregated, horizon)
        objective_value = metric_set.get(objective_metric)
        elapsed = time.perf_counter() - t0

        if objective_value is None:
            trial_results.append(
                _failed_trial_row(trial_id, config, use_exog, objective_metric,
                                  len(full_y), elapsed, "all rolling-origin cycles failed")
            )
            raise RuntimeError(f"{trial_id}: all rolling-origin cycles failed")

        # Ljung-Box eligibility on the last rolling-origin cycle.
        lb_pvalue, excluded, lb_cycle_index = _ljung_box_eligibility_on_last_cycle(
            config=config,
            full_df=full_df,
            cycle=cycles[-1],
            date_col=date_col,
            target_col=target_col,
            exog_cols=exog_cols,
            use_exog=use_exog,
            lags=lb_lags,
            threshold=lb_threshold,
            enabled=lb_enabled,
        )
        full_result = _fit_sarimax_result(
            config=config,
            endog=full_y,
            exog=full_exog,
            use_exog=use_exog,
            context="full-history candidate refit",
        )
        if full_result is None:
            trial_results.append(
                _failed_trial_row(
                    trial_id,
                    config,
                    use_exog,
                    objective_metric,
                    len(full_y),
                    elapsed,
                    "full-history candidate refit failed",
                )
            )
            raise RuntimeError(f"{trial_id}: full-history candidate refit failed")
        full_history_fits[trial_id] = full_result

        trial_results.append(
            {
                "trial_id": trial_id,
                "status": "success",
                "model_family": "sarimax",
                "granularity": "monthly",
                "order": config["order"],
                "seasonal_order": config["seasonal_order"],
                "trend": config.get("trend"),
                "enforce_stationarity": config.get("enforce_stationarity", False),
                "enforce_invertibility": config.get("enforce_invertibility", False),
                "use_exog": use_exog,
                "objective_metric": objective_metric,
                "objective_value": objective_value,
                **metric_set,
                "ljung_box_pvalue": lb_pvalue,
                "autocorrelation_excluded": excluded,
                "ljung_box_cycle_index": lb_cycle_index,
                "n_full_history": len(full_y),
                "fit_seconds": round(elapsed, 3),
                "error_message": None,
            }
        )
        logger.info(
            "%s ✓  wmape=%.4f  wmape_m1=%.4f  wmape_m2=%.4f  wmape_m3=%.4f  bias=%.4f  lb_p=%s  excluded=%s",
            trial_id,
            metric_set["wmape"],
            metric_set.get("wmape_m1") or float("nan"),
            metric_set.get("wmape_m2") or float("nan"),
            metric_set["wmape_m3"],
            metric_set["bias"],
            f"{lb_pvalue:.4f}" if lb_pvalue is not None else "nan", excluded,
        )
        log_trial_predictions(trial_id, _trial_preds)
        return float(objective_value)

    callbacks = [_stop_on_max_failures]
    study.optimize(objective, n_trials=max_trials, catch=(Exception,), callbacks=callbacks)

    successful = [r for r in trial_results if r["status"] == "success"]
    failed = [r for r in trial_results if r["status"] == "failed"]
    if not successful:
        summaries = [f"{r['trial_id']}: {r['error_message']}" for r in failed]
        raise RuntimeError(
            "All SARIMAX trials failed during rolling-origin evaluation.\n"
            + "\n".join(summaries)
        )

    # ── rank and select eligible pre-champions ────────────────────────────────
    tuning_df = _rank_candidates(
        pd.DataFrame(trial_results), objective_metric, objective_direction
    )
    prechampion_ids = _select_prechampion_ids(tuning_df, top_n)
    logger.info("Pre-champion SARIMAX candidates (top-%d eligible): %s", top_n, prechampion_ids)

    metric_cols = ["wmape", "wmape_m1", "wmape_m2", "wmape_m3", "mase", "bias", "rmse"]
    ro_metrics_df = (
        tuning_df[tuning_df["status"] == "success"][
            ["trial_id", "rank", "model_family", "granularity", "order",
             "seasonal_order", "trend", "use_exog", *metric_cols,
             "ljung_box_pvalue", "autocorrelation_excluded",
             "ljung_box_cycle_index", "n_full_history"]
        ]
        .assign(seasonal_period=mase_seasonal_period)
        .reset_index(drop=True)
    )

    prechampion_configs = _build_prechampion_configs(
        tuning_df, prechampion_ids, objective_metric, top_n, mase_seasonal_period, exog_cols
    )

    # ── persist top-N full-history-refit models ───────────────────────────────
    candidate_models: dict[str, Any] = {}
    for cid in prechampion_ids:
        if cid not in full_history_fits:
            continue
        row = tuning_df[tuning_df["trial_id"] == cid].iloc[0]
        candidate_models[cid] = {
            "rank": _safe_int(row["rank"]),
            "model_family": "sarimax",
            "granularity": "monthly",
            "config": {
                "order": row["order"],
                "seasonal_order": row["seasonal_order"],
                "trend": row["trend"],
                "use_exog": row["use_exog"],
            },
            "exogenous_columns": exog_cols if use_exog else [],
            "model": full_history_fits[cid],
            "rolling_origin_metrics": _metric_set_from_row(row),
            "ljung_box_pvalue": _safe_float(row["ljung_box_pvalue"]),
            "autocorrelation_excluded": bool(row["autocorrelation_excluded"]),
            "ljung_box_cycle_index": _safe_int(row["ljung_box_cycle_index"]),
        }

    run_ts = datetime.now(tz=UTC).isoformat()
    best_row = (
        tuning_df[tuning_df["rank"] == 1].iloc[0]
        if (tuning_df["rank"] == 1).any()
        else None
    )
    training_metadata = _build_training_metadata(
        trial_results=trial_results, study=study, best_row=best_row,
        objective_metric=objective_metric, objective_direction=objective_direction,
        top_n=top_n, mase_seasonal_period=mase_seasonal_period, use_exog=use_exog,
        exog_cols=exog_cols, target_col=target_col, date_col=date_col,
        full_df=full_df, run_ts=run_ts, prechampion_ids=prechampion_ids,
        rolling_origin_cfg=rolling_origin_cfg,
    )

    if not prechampion_ids or prechampion_ids[0] not in candidate_models:
        raise RuntimeError("Rank-1 SARIMAX candidate model was not persisted correctly.")

    rank1_id = prechampion_ids[0]
    rank1_row = tuning_df[tuning_df["trial_id"] == rank1_id].iloc[0]
    candidate_monthly_sarimax: dict = {
        "model_family": "sarimax",
        "granularity": "monthly",
        "candidate_id": rank1_id,
        "rank": 1,
        "config": {
            "order": rank1_row["order"],
            "seasonal_order": rank1_row["seasonal_order"],
            "trend": rank1_row["trend"],
            "use_exog": rank1_row["use_exog"],
        },
        "exogenous_columns": exog_cols if use_exog else [],
        "model": full_history_fits[rank1_id],
        "rolling_origin_metrics": _metric_set_from_row(rank1_row),
        "metadata": training_metadata,
    }

    logger.info(
        "SARIMAX training done — trials=%d  successful=%d  failed=%d  best=%s  wmape_m3=%.4f",
        len(trial_results), len(successful), len(failed), rank1_id,
        _safe_float(rank1_row["wmape_m3"]) or float("nan"),
    )

    return (
        tuning_df,
        ro_metrics_df,
        prechampion_configs,
        candidate_models,
        training_metadata,
        candidate_monthly_sarimax,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_sarimax_training_inputs(
    full_df: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> None:
    """Raise a descriptive error for an invalid full-history input."""
    if full_df.empty:
        raise ValueError("SARIMAX full-history DataFrame is empty.")
    for col in (date_col, target_col):
        if col not in full_df.columns:
            raise ValueError(f"Required column {col!r} missing from full-history data.")
    if not pd.api.types.is_numeric_dtype(full_df[target_col]):
        raise ValueError(
            f"Target column {target_col!r} must be numeric; "
            f"found dtype {full_df[target_col].dtype}."
        )


def _extract_exog(df: pd.DataFrame, exog_cols: list[str]) -> np.ndarray | None:
    """Return an exogenous matrix from the DataFrame, or None if no columns given."""
    if not exog_cols:
        return None
    missing = [c for c in exog_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Exogenous columns missing from DataFrame: {missing}")
    return df[exog_cols].values.astype(float)


def _build_sarimax_config_from_trial(params: dict, s: int) -> dict:
    """Assemble a SARIMAX config dict from flat Optuna trial params.

    Maps the string sentinel ``"none"`` back to Python ``None`` for the trend
    argument, since Optuna cannot serialise ``None`` in categorical search spaces.
    """
    p = int(params["p"])
    d = int(params["d"])
    q = int(params["q"])
    big_p = int(params["P"])
    big_d = int(params["D"])
    big_q = int(params["Q"])
    trend_raw = params.get("trend", "none")
    trend = None if str(trend_raw).lower() == "none" else str(trend_raw)
    return {
        "order": [p, d, q],
        "seasonal_order": [big_p, big_d, big_q, s],
        "trend": trend,
        "enforce_stationarity": bool(params.get("enforce_stationarity", False)),
        "enforce_invertibility": bool(params.get("enforce_invertibility", False)),
    }


def _sarimax_cycle_forecast(  # noqa: PLR0913
    config: dict,
    train_df: pd.DataFrame,
    cycle: RollingOriginCycle,
    full_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    exog_cols: list[str],
    use_exog: bool,
) -> np.ndarray:
    """Refit SARIMAX on a cycle's train window and forecast its target months.

    Future exogenous values for the (observed) target months come from the
    full-history frame, mirroring production multi-step forecasting.

    Returns:
        Array of ``len(cycle.target_dates)`` point forecasts in chronological order.
    """
    train_y = train_df[target_col].to_numpy(dtype=float)
    train_exog = _extract_exog(train_df, exog_cols) if use_exog else None

    target_set = pd.DatetimeIndex(cycle.target_dates)
    target_rows = full_df[full_df[date_col].isin(target_set)].sort_values(date_col)
    future_exog = _extract_exog(target_rows, exog_cols) if use_exog else None

    model = SARIMAX(
        endog=train_y,
        exog=train_exog,
        order=tuple(config["order"]),
        seasonal_order=tuple(config["seasonal_order"]),
        trend=config.get("trend"),
        enforce_stationarity=bool(config.get("enforce_stationarity", False)),
        enforce_invertibility=bool(config.get("enforce_invertibility", False)),
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        result = model.fit(disp=False)

    steps = len(cycle.target_dates)
    forecast_out = result.get_forecast(steps=steps, exog=future_exog)
    y_pred = np.asarray(forecast_out.predicted_mean, dtype=float)
    if not np.isfinite(y_pred).all():
        raise ValueError("SARIMAX forecast produced non-finite value(s).")
    return y_pred


def _fit_sarimax_result(  # noqa: PLR0913
    config: dict,
    endog: np.ndarray,
    exog: np.ndarray | None,
    use_exog: bool,
    context: str,
) -> Any | None:
    """Fit a SARIMAX result object, returning ``None`` on model-fit failure."""
    try:
        model = SARIMAX(
            endog=endog,
            exog=exog if use_exog else None,
            order=tuple(config["order"]),
            seasonal_order=tuple(config["seasonal_order"]),
            trend=config.get("trend"),
            enforce_stationarity=bool(config.get("enforce_stationarity", False)),
            enforce_invertibility=bool(config.get("enforce_invertibility", False)),
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            return model.fit(disp=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SARIMAX %s failed: %s", context, exc)
        return None


def _ljung_box_eligibility_on_last_cycle(  # noqa: PLR0913
    config: dict,
    full_df: pd.DataFrame,
    cycle: RollingOriginCycle,
    date_col: str,
    target_col: str,
    exog_cols: list[str],
    use_exog: bool,
    lags: int,
    threshold: float,
    enabled: bool,
) -> tuple[float | None, bool, int | None]:
    """Assess Ljung-Box residual eligibility on the last rolling-origin cycle.

    Returns:
        Tuple ``(pvalue, excluded, cycle_index)``. ``excluded`` is True when the
        residuals show significant autocorrelation (p < threshold) and the filter
        is enabled. The diagnostic is calculated on the same train window used by
        the last rolling-origin cycle, not on the production full-history refit.
    """
    if not enabled:
        return None, False, cycle.cycle_index

    train_df = full_df[full_df[date_col] <= cycle.origin_date].copy()
    train_y = train_df[target_col].to_numpy(dtype=float)
    train_exog = _extract_exog(train_df, exog_cols) if use_exog else None
    result = _fit_sarimax_result(
        config=config,
        endog=train_y,
        exog=train_exog,
        use_exog=use_exog,
        context=f"Ljung-Box cycle {cycle.cycle_index} fit",
    )
    if result is None:
        return None, True, cycle.cycle_index

    try:
        lb_df = acorr_ljungbox(result.resid, lags=[lags], return_df=True)
        pvalue = float(lb_df["lb_pvalue"].iloc[-1])
        excluded = pvalue < threshold
        return pvalue, bool(excluded), cycle.cycle_index
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ljung-Box test failed (%s) — keeping candidate eligible.", exc)
        return None, False, cycle.cycle_index


def _failed_trial_row(  # noqa: PLR0913
    trial_id: str,
    config: dict,
    use_exog: bool,
    objective_metric: str,
    n_full: int,
    elapsed: float,
    error: str,
) -> dict:
    """Build a failed-trial record with the rolling-origin metric columns nulled."""
    null_metrics = {
        k: None
        for k in ["wmape", "wmape_m1", "wmape_m2", "wmape_m3", "mase", "bias", "rmse"]
    }
    return {
        "trial_id": trial_id,
        "status": "failed",
        "model_family": "sarimax",
        "granularity": "monthly",
        "order": config["order"],
        "seasonal_order": config["seasonal_order"],
        "trend": config.get("trend"),
        "enforce_stationarity": config.get("enforce_stationarity", False),
        "enforce_invertibility": config.get("enforce_invertibility", False),
        "use_exog": use_exog,
        "objective_metric": objective_metric,
        "objective_value": None,
        **null_metrics,
        "ljung_box_pvalue": None,
        "autocorrelation_excluded": False,
        "ljung_box_cycle_index": None,
        "n_full_history": n_full,
        "fit_seconds": round(elapsed, 3),
        "error_message": error,
    }


def _rank_candidates(
    df: pd.DataFrame, objective_metric: str, objective_direction: str
) -> pd.DataFrame:
    """Rank successful candidates by objective metric; append failed rows at the end."""
    success_mask = df["status"] == "success"
    success_df = df[success_mask].copy()
    failed_df = df[~success_mask].copy()

    ascending = objective_direction == "minimize"
    success_df = success_df.sort_values(
        "objective_value", ascending=ascending, na_position="last"
    ).reset_index(drop=True)
    success_df["rank"] = list(range(1, len(success_df) + 1))
    failed_df["rank"] = None
    return pd.concat([success_df, failed_df], ignore_index=True)


def _select_prechampion_ids(tuning_df: pd.DataFrame, top_n: int) -> list[str]:
    """Select the top-N eligible pre-champion trial ids (Ljung-Box aware).

    Prefers candidates not flagged by the Ljung-Box filter; falls back to all
    successful candidates (ordered by rank) when none are eligible.
    """
    success = tuning_df[tuning_df["status"] == "success"].copy()
    success = success[success["rank"].notna()].sort_values("rank")

    eligible = success[~success["autocorrelation_excluded"].fillna(False)]
    pool = eligible if not eligible.empty else success
    if eligible.empty and not success.empty:
        logger.warning(
            "All SARIMAX candidates were Ljung-Box excluded; falling back to the "
            "full ranking for the pre-champion shortlist."
        )
    return pool["trial_id"].head(top_n).tolist()


def _metric_set_from_row(row: "pd.Series") -> dict:
    """Extract the rolling-origin metric set from a tuning row."""
    keys = ["wmape", "wmape_m1", "wmape_m2", "wmape_m3", "mase", "bias", "rmse"]
    return {k: _safe_float(row.get(k)) for k in keys}


def _build_prechampion_configs(  # noqa: PLR0913
    ranked_df: pd.DataFrame,
    prechampion_ids: list[str],
    objective_metric: str,
    top_n: int,
    seasonal_period: int,
    exog_cols: list[str],
) -> dict:
    """Build the prechampion_configs JSON artifact."""
    candidates = []
    for rank_pos, trial_id in enumerate(prechampion_ids, start=1):
        row = ranked_df[ranked_df["trial_id"] == trial_id]
        if row.empty:
            continue
        r = row.iloc[0]
        use_exog_entry = bool(r["use_exog"])
        candidates.append(
            {
                "trial_id": trial_id,
                "rank": rank_pos,
                "order": r["order"],
                "seasonal_order": r["seasonal_order"],
                "trend": r["trend"],
                "use_exog": use_exog_entry,
                "exogenous_columns": exog_cols if use_exog_entry else [],
                "rolling_origin_metrics": _metric_set_from_row(r),
                "ljung_box_pvalue": _safe_float(r["ljung_box_pvalue"]),
                "autocorrelation_excluded": bool(r["autocorrelation_excluded"]),
                "ljung_box_cycle_index": _safe_int(r["ljung_box_cycle_index"]),
            }
        )

    return {
        "model_family": "sarimax",
        "granularity": "monthly",
        "selection_stage": "rolling_origin",
        "objective_metric": objective_metric,
        "top_n": top_n,
        "seasonal_period": seasonal_period,
        "candidates": candidates,
    }


def _build_training_metadata(  # noqa: PLR0913
    trial_results: list[dict],
    study: optuna.Study,
    best_row: "pd.Series | None",
    objective_metric: str,
    objective_direction: str,
    top_n: int,
    mase_seasonal_period: int,
    use_exog: bool,
    exog_cols: list[str],
    target_col: str,
    date_col: str,
    full_df: pd.DataFrame,
    run_ts: str,
    prechampion_ids: list[str] | None = None,
    rolling_origin_cfg: dict | None = None,
) -> dict:
    """Build the training_metadata JSON artifact."""
    successful = [r for r in trial_results if r["status"] == "success"]
    failed = [r for r in trial_results if r["status"] == "failed"]
    ro_cfg = rolling_origin_cfg or {}

    best_trial_id = str(best_row["trial_id"]) if best_row is not None else None
    best_val = _safe_float(best_row[objective_metric]) if best_row is not None else None

    n_completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )

    warnings_list: list[str] = []
    if len(successful) < top_n:
        warnings_list.append(
            f"Only {len(successful)} successful trial(s); requested top_n={top_n}."
        )

    return {
        "model_family": "sarimax",
        "model_class": "statsmodels.tsa.statespace.sarimax.SARIMAX",
        "granularity": "monthly",
        "training_stage": "rolling_origin",
        "evaluation_mode": "rolling_origin",
        "run_timestamp": run_ts,
        "optimizer": "optuna",
        "study_direction": objective_direction,
        "objective_metric": objective_metric,
        "objective_direction": objective_direction,
        "rolling_origin": {
            "horizon": int(ro_cfg.get("horizon", 3)),
            "n_cycles": int(ro_cfg.get("n_cycles", 5)),
            "window": str(ro_cfg.get("window", "expanding")),
            "step_months": int(ro_cfg.get("step_months", 1)),
        },
        "top_n_prechampions": top_n,
        "n_trials_configured": len(study.trials),
        "n_trials_attempted": len(trial_results),
        "n_trials_completed": n_completed,
        "n_trials_pruned": n_pruned,
        "n_trials_successful": len(successful),
        "n_trials_failed": len(failed),
        "best_trial_id": best_trial_id,
        "best_rank": 1 if best_row is not None else None,
        "best_objective_value": best_val,
        "seasonal_period": mase_seasonal_period,
        "uses_exogenous_features": use_exog,
        "exogenous_columns": exog_cols,
        "exogenous_column_order": exog_cols,
        "selected_prechampion_ids": list(prechampion_ids) if prechampion_ids else [],
        "target_column": target_col,
        "date_column": date_col,
        "full_history_start": str(full_df[date_col].min().date()),
        "full_history_end": str(full_df[date_col].max().date()),
        "warnings": warnings_list,
        "failed_trials": [
            {"trial_id": r["trial_id"], "error": r["error_message"]} for r in failed
        ],
    }


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
