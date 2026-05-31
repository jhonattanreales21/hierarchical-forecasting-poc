"""Monthly SARIMAX training nodes: controlled grid search, validation, and artifact emission."""

import itertools
import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from shared.metrics import mase as _shared_mase
from shared.metrics import wape as _shared_wape
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


# ── Public node ───────────────────────────────────────────────────────────────


def train_and_evaluate_monthly_sarimax_candidates(
    monthly_sarimax_train: pd.DataFrame,
    monthly_sarimax_validation: pd.DataFrame,
    monthly_sarimax_split_metadata: dict,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict]:
    """Train, validate, rank, and emit monthly SARIMAX candidate artifacts.

    Runs a deterministic grid search over the configured SARIMAX parameter space.
    Each configuration is fit on the training split and evaluated on the validation
    split. Successful candidates are ranked by the configured objective metric and
    the top-N are persisted as pre-champion artifacts.

    Args:
        monthly_sarimax_train: SARIMAX-ready training DataFrame with date and target columns.
        monthly_sarimax_validation: SARIMAX-ready validation DataFrame with the same schema.
        monthly_sarimax_split_metadata: Metadata from the SARIMAX input adapter (date ranges,
            column names, exogenous columns).
        params: Contents of ``train_monthly.sarimax`` from the parameter file.

    Returns:
        Six-element tuple:

        1. ``tuning_results`` — DataFrame, one row per trial (ranked).
        2. ``validation_metrics`` — DataFrame, detailed per-candidate metrics.
        3. ``prechampion_configs`` — Dict with top-N pre-champion configurations.
        4. ``candidate_models`` — Dict mapping trial_id → fitted SARIMAX result (top-N).
        5. ``training_metadata`` — Dict summarizing the grid search run.
        6. ``candidate_monthly_sarimax`` — Rank-1 SARIMAX candidate dict.

    Raises:
        ValueError: When inputs are invalid or required columns are missing.
        RuntimeError: When every configured trial fails to train.
    """
    # ── extract configuration ─────────────────────────────────────────────────
    date_col: str = params.get("date_column", "month_start_date")
    target_col: str = params.get("target_column", "monthly_demand")
    objective_cfg: dict = params.get("objective", {})
    objective_metric: str = str(objective_cfg.get("metric", "wape"))
    objective_direction: str = str(objective_cfg.get("direction", "minimize")).lower()
    top_n: int = int(params.get("top_n_prechampions", 3))
    metrics_cfg: dict = params.get("metrics", {})
    mase_seasonal_period: int = int(metrics_cfg.get("mase_seasonal_period", 12))
    epsilon: float = float(metrics_cfg.get("epsilon", 1.0))
    tuning_cfg: dict = params.get("tuning", {})
    use_exog: bool = bool(tuning_cfg.get("use_exog", True))

    # ── validate inputs ───────────────────────────────────────────────────────
    _validate_sarimax_training_inputs(
        monthly_sarimax_train, monthly_sarimax_validation, date_col, target_col
    )

    # ── extract series and exogenous matrices ─────────────────────────────────
    exog_cols: list[str] = list(monthly_sarimax_split_metadata.get("exogenous_columns", []))
    if use_exog and not exog_cols:
        logger.info(
            "use_exog=True but no exogenous columns in split metadata — running without exog."
        )
        use_exog = False

    train_df = monthly_sarimax_train.copy()
    val_df = monthly_sarimax_validation.copy()
    train_df[date_col] = pd.to_datetime(train_df[date_col])
    val_df[date_col] = pd.to_datetime(val_df[date_col])
    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    val_df = val_df.sort_values(date_col).reset_index(drop=True)

    train_y = train_df[target_col].values.astype(float)
    val_y = val_df[target_col].values.astype(float)
    train_exog = _extract_exog(train_df, exog_cols) if use_exog else None
    val_exog = _extract_exog(val_df, exog_cols) if use_exog else None

    forecast_steps = int(params.get("validation", {}).get("forecast_steps") or len(val_y))

    logger.info(
        "SARIMAX training — train=%d rows  val=%d rows  exog_cols=%s  "
        "objective=%s  direction=%s",
        len(train_y),
        len(val_y),
        exog_cols if use_exog else "(none)",
        objective_metric,
        objective_direction,
    )

    # ── build parameter grid ──────────────────────────────────────────────────
    param_grid = _build_param_grid(tuning_cfg)
    logger.info("SARIMAX grid search: %d configurations to try.", len(param_grid))

    # ── run grid search ───────────────────────────────────────────────────────
    trial_results: list[dict] = []
    fitted_models: dict[str, Any] = {}

    for idx, config in enumerate(param_grid):
        trial_id = f"sarimax_trial_{idx + 1:03d}"
        t0 = time.perf_counter()

        logger.info(
            "[%d/%d] %s order=%s seasonal_order=%s trend=%s use_exog=%s",
            idx + 1,
            len(param_grid),
            trial_id,
            config["order"],
            config["seasonal_order"],
            config.get("trend"),
            use_exog,
        )

        try:
            model = SARIMAX(
                endog=train_y,
                exog=train_exog,
                order=tuple(config["order"]),
                seasonal_order=tuple(config["seasonal_order"]),
                trend=config.get("trend"),
                enforce_stationarity=bool(config.get("enforce_stationarity", False)),
                enforce_invertibility=bool(config.get("enforce_invertibility", False)),
            )
            result = model.fit(disp=False)

            forecast_out = result.get_forecast(steps=forecast_steps, exog=val_exog)
            y_pred = np.asarray(forecast_out.predicted_mean, dtype=float)

            # Align shapes: forecast may produce more steps than validation rows
            n_eval = min(len(y_pred), len(val_y))
            metrics = _compute_validation_metrics(
                val_y[:n_eval], y_pred[:n_eval], train_y, mase_seasonal_period, epsilon
            )
            elapsed = time.perf_counter() - t0

            fitted_models[trial_id] = result
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
                    "objective_value": metrics[objective_metric],
                    **metrics,
                    "n_train": len(train_y),
                    "n_validation": n_eval,
                    "train_start": str(train_df[date_col].min().date()),
                    "train_end": str(train_df[date_col].max().date()),
                    "validation_start": str(val_df[date_col].min().date()),
                    "validation_end": str(val_df[date_col].max().date()),
                    "fit_seconds": round(elapsed, 3),
                    "error_message": None,
                }
            )
            logger.info(
                "%s ✓  wape=%.4f  mase=%.4f  rmse=%.2f  bias=%.4f",
                trial_id,
                metrics["wape"],
                metrics["mase"] if metrics["mase"] is not None else float("nan"),
                metrics["rmse"],
                metrics["bias"],
            )

        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            logger.warning(
                "%s FAILED (%.2fs): %s | config=%s", trial_id, elapsed, exc, config
            )
            trial_results.append(
                {
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
                    "wape": None,
                    "mase": None,
                    "rmse": None,
                    "bias": None,
                    "mae": None,
                    "n_train": len(train_y),
                    "n_validation": len(val_y),
                    "train_start": str(train_df[date_col].min().date()),
                    "train_end": str(train_df[date_col].max().date()),
                    "validation_start": str(val_df[date_col].min().date()),
                    "validation_end": str(val_df[date_col].max().date()),
                    "fit_seconds": round(elapsed, 3),
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )

    # ── validate at least one success ─────────────────────────────────────────
    successful = [r for r in trial_results if r["status"] == "success"]
    failed = [r for r in trial_results if r["status"] == "failed"]
    if not successful:
        summaries = [
            f"{r['trial_id']}: {r['error_message']}" for r in failed
        ]
        raise RuntimeError(
            "All SARIMAX trials failed during monthly training.\n"
            + "\n".join(summaries)
        )

    # ── rank and select pre-champions ─────────────────────────────────────────
    tuning_df = _rank_candidates(pd.DataFrame(trial_results), objective_metric, objective_direction)

    ranked_mask = tuning_df["rank"].notna()
    # Cast rank to float for sorting safety (nullable int may have object dtype after concat)
    prechampion_ids: list[str] = (
        tuning_df[ranked_mask]
        .assign(_rank_float=tuning_df.loc[ranked_mask, "rank"].astype(float))
        .nsmallest(top_n, "_rank_float")["trial_id"]
        .tolist()
    )
    logger.info("Pre-champion SARIMAX candidates (top-%d): %s", top_n, prechampion_ids)

    # ── build validation metrics table (successful candidates only) ───────────
    val_metrics_df = tuning_df[tuning_df["status"] == "success"][
        [
            "trial_id",
            "rank",
            "model_family",
            "granularity",
            "order",
            "seasonal_order",
            "trend",
            "use_exog",
            "wape",
            "mase",
            "rmse",
            "bias",
            "mae",
            "n_train",
            "n_validation",
            "validation_start",
            "validation_end",
        ]
    ].assign(seasonal_period=mase_seasonal_period).reset_index(drop=True)

    # ── build prechampion_configs artifact ────────────────────────────────────
    prechampion_configs = _build_prechampion_configs(
        tuning_df, prechampion_ids, objective_metric, top_n, mase_seasonal_period
    )

    # ── persist top-N fitted models ───────────────────────────────────────────
    candidate_models: dict[str, Any] = {}
    for cid in prechampion_ids:
        if cid not in fitted_models:
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
            "model": fitted_models[cid],
            "validation_metrics": {
                "wape": _safe_float(row["wape"]),
                "mase": _safe_float(row["mase"]),
                "rmse": _safe_float(row["rmse"]),
                "bias": _safe_float(row["bias"]),
                "mae": _safe_float(row["mae"]),
            },
        }

    # ── build training metadata ───────────────────────────────────────────────
    run_ts = datetime.now(tz=UTC).isoformat()
    best_row = tuning_df[tuning_df["rank"] == 1].iloc[0] if (tuning_df["rank"] == 1).any() else None
    training_metadata = _build_training_metadata(
        trial_results=trial_results,
        best_row=best_row,
        objective_metric=objective_metric,
        objective_direction=objective_direction,
        top_n=top_n,
        mase_seasonal_period=mase_seasonal_period,
        use_exog=use_exog,
        exog_cols=exog_cols,
        target_col=target_col,
        date_col=date_col,
        train_df=train_df,
        val_df=val_df,
        run_ts=run_ts,
    )

    # ── build rank-1 candidate artifact ──────────────────────────────────────
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
        "model": fitted_models[rank1_id],
        "validation_metrics": {
            "wape": _safe_float(rank1_row["wape"]),
            "mase": _safe_float(rank1_row["mase"]),
            "rmse": _safe_float(rank1_row["rmse"]),
            "bias": _safe_float(rank1_row["bias"]),
            "mae": _safe_float(rank1_row["mae"]),
        },
        "metadata": training_metadata,
    }

    logger.info(
        "SARIMAX training done — trials=%d  successful=%d  failed=%d  "
        "best=%s  wape=%.4f",
        len(trial_results),
        len(successful),
        len(failed),
        rank1_id,
        _safe_float(rank1_row["wape"]) or float("nan"),
    )

    return (
        tuning_df,
        val_metrics_df,
        prechampion_configs,
        candidate_models,
        training_metadata,
        candidate_monthly_sarimax,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_sarimax_training_inputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> None:
    """Raise a descriptive error for invalid training or validation inputs."""
    if train_df.empty:
        raise ValueError("SARIMAX training DataFrame is empty.")
    if val_df.empty:
        raise ValueError("SARIMAX validation DataFrame is empty.")
    for col in (date_col, target_col):
        if col not in train_df.columns:
            raise ValueError(f"Required column {col!r} missing from training data.")
        if col not in val_df.columns:
            raise ValueError(f"Required column {col!r} missing from validation data.")
    if not pd.api.types.is_numeric_dtype(train_df[target_col]):
        raise ValueError(
            f"Target column {target_col!r} must be numeric; "
            f"found dtype {train_df[target_col].dtype}."
        )


def _extract_exog(df: pd.DataFrame, exog_cols: list[str]) -> np.ndarray | None:
    """Return an exogenous matrix from the DataFrame, or None if no columns given."""
    if not exog_cols:
        return None
    missing = [c for c in exog_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Exogenous columns missing from DataFrame: {missing}")
    return df[exog_cols].values.astype(float)


def _build_param_grid(tuning_cfg: dict) -> list[dict]:
    """Build the deterministic SARIMAX configuration grid from tuning config."""
    order_grid = [list(o) for o in tuning_cfg.get("order_grid", [[1, 1, 1]])]
    seasonal_order_grid = [
        list(so) for so in tuning_cfg.get("seasonal_order_grid", [[0, 1, 1, 12]])
    ]
    trend_options: list = tuning_cfg.get("trend_options", [None])
    enforce_stationarity_options: list = tuning_cfg.get(
        "enforce_stationarity_options", [False]
    )
    enforce_invertibility_options: list = tuning_cfg.get(
        "enforce_invertibility_options", [False]
    )

    grid = []
    for order, seasonal_order, trend, enforce_stat, enforce_inv in itertools.product(
        order_grid,
        seasonal_order_grid,
        trend_options,
        enforce_stationarity_options,
        enforce_invertibility_options,
    ):
        grid.append(
            {
                "order": order,
                "seasonal_order": seasonal_order,
                "trend": trend,
                "enforce_stationarity": enforce_stat,
                "enforce_invertibility": enforce_inv,
            }
        )
    return grid


def _compute_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    mase_seasonal_period: int,
    epsilon: float,
) -> dict:
    """Compute WAPE, MASE, RMSE, bias, and MAE for a single SARIMAX candidate."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    wape_val = _shared_wape(y_true, y_pred)
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae_val = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum(np.abs(y_true)))
    bias_val = float(np.sum(y_pred - y_true)) / (denom + epsilon)

    mase_val: float | None = None
    if y_train is not None and len(y_train) > mase_seasonal_period:
        raw = _shared_mase(y_true, y_pred, y_train, mase_seasonal_period)
        mase_val = raw if (raw is not None and np.isfinite(raw)) else None

    return {
        "wape": wape_val,
        "mase": mase_val,
        "rmse": rmse_val,
        "bias": bias_val,
        "mae": mae_val,
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


def _build_prechampion_configs(
    ranked_df: pd.DataFrame,
    prechampion_ids: list[str],
    objective_metric: str,
    top_n: int,
    seasonal_period: int,
) -> dict:
    """Build the prechampion_configs JSON artifact."""
    candidates = []
    for rank_pos, trial_id in enumerate(prechampion_ids, start=1):
        row = ranked_df[ranked_df["trial_id"] == trial_id]
        if row.empty:
            continue
        r = row.iloc[0]
        candidates.append(
            {
                "trial_id": trial_id,
                "rank": rank_pos,
                "order": r["order"],
                "seasonal_order": r["seasonal_order"],
                "trend": r["trend"],
                "use_exog": bool(r["use_exog"]),
                "metrics": {
                    "wape": _safe_float(r["wape"]),
                    "mase": _safe_float(r["mase"]),
                    "rmse": _safe_float(r["rmse"]),
                    "bias": _safe_float(r["bias"]),
                    "mae": _safe_float(r["mae"]),
                },
            }
        )

    return {
        "model_family": "sarimax",
        "granularity": "monthly",
        "selection_stage": "validation",
        "objective_metric": objective_metric,
        "top_n": top_n,
        "seasonal_period": seasonal_period,
        "candidates": candidates,
    }


def _build_training_metadata(  # noqa: PLR0913
    trial_results: list[dict],
    best_row: "pd.Series | None",
    objective_metric: str,
    objective_direction: str,
    top_n: int,
    mase_seasonal_period: int,
    use_exog: bool,
    exog_cols: list[str],
    target_col: str,
    date_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_ts: str,
) -> dict:
    """Build the training_metadata JSON artifact."""
    successful = [r for r in trial_results if r["status"] == "success"]
    failed = [r for r in trial_results if r["status"] == "failed"]

    best_trial_id = str(best_row["trial_id"]) if best_row is not None else None
    best_val = _safe_float(best_row[objective_metric]) if best_row is not None else None

    warnings_list: list[str] = []
    if len(successful) < top_n:
        warnings_list.append(
            f"Only {len(successful)} successful trial(s); requested top_n={top_n}."
        )

    return {
        "model_family": "sarimax",
        "granularity": "monthly",
        "training_stage": "validation",
        "run_timestamp": run_ts,
        "objective_metric": objective_metric,
        "objective_direction": objective_direction,
        "top_n_prechampions": top_n,
        "n_trials_configured": len(trial_results),
        "n_trials_attempted": len(trial_results),
        "n_trials_successful": len(successful),
        "n_trials_failed": len(failed),
        "best_trial_id": best_trial_id,
        "best_rank": 1 if best_row is not None else None,
        "best_validation_metric": best_val,
        "seasonal_period": mase_seasonal_period,
        "uses_exogenous_features": use_exog,
        "exogenous_columns": exog_cols,
        "target_column": target_col,
        "date_column": date_col,
        "train_start": str(train_df[date_col].min().date()),
        "train_end": str(train_df[date_col].max().date()),
        "validation_start": str(val_df[date_col].min().date()),
        "validation_end": str(val_df[date_col].max().date()),
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
